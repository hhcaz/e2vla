import os
import sys
import tyro
import torch
import envars
import argparse
import torch.amp
from typing import Dict
from datetime import datetime
from torch import Tensor, optim
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter

from models import vla
from configs import CONFIGS, TrainConfig
from train_utils.ema_impl import ExponentialMovingAverage
from data_utils.dataset_base import get_dataloader, generate_sample_weights


def init_train_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-s", dest="save", type=str, default="", help="exp name to save")
    parser.add_argument("-c", dest="conti", type=str, default="", help="exp name to resume")

    if "-h" in sys.argv or "--help" in sys.argv:
        print("=== argparse help ===")
        parser.print_help()
        print("\n=== tyro help ===")
        tyro.extras.get_parser(TrainConfig).print_help()
        sys.exit(0)

    args, remaining_argv = parser.parse_known_args()
    save: str = args.save
    conti: str = args.conti

    cfg = CONFIGS[args.config]
    cfg = tyro.cli(cfg.__class__, default=cfg, args=remaining_argv)
    return cfg, save, conti


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.sum = 0
        self.count = 0
    
    def append(self, val):
        self.sum += val
        self.count += 1
    
    def avg(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count


def count_trainable(m: torch.nn.Module):
    count = 0
    for p in m.parameters():
        if p.requires_grad:
            count += p.numel()
    return count


def get_data_loader_for_cfg(cfg: TrainConfig):
    if cfg.sample_multiplex > 1:
        assert cfg.dataset_weights is not None, \
            "sample_multiplex should be used together with dataset_weights"

    datasets = [D.inst() for D in cfg.dataset_classes]
    if cfg.dataset_weights is not None:
        sample_weights = generate_sample_weights(datasets, cfg.dataset_weights)
    else:
        sample_weights = None
    
    shuffle = True if (cfg.dataset_weights is None) else None
    dataloader = get_dataloader(
        datasets=datasets,
        batch_size=cfg.bs,
        num_workers=cfg.workers,
        shuffle=shuffle,
        persistent_workers=True,
        sample_weights=sample_weights,
        sample_multiplex=cfg.sample_multiplex
    )
    return dataloader


class Trainer(object):
    LOG_DIR = "./logs/E2VLA"
    CKPT_DIR = "./checkpoints/E2VLA"

    def __init__(self):

        self.launch_time_str = datetime.now().strftime("%Y%m%d%H%M")
        self.cfg, save, conti = init_train_config()
        print("[INFO] Train config:")
        print(self.cfg)

        self.model_device = "cuda:0"
        self.model: vla.VLA = getattr(vla, "vla_" + self.cfg.model.strip()
                                      )().to(self.model_device)

        print("[INFO] Total {:.3f}M trainable parameters"
              .format(count_trainable(self.model) / 1e6))
        
        self.train_loader = get_data_loader_for_cfg(self.cfg)
        self.scaler = torch.amp.GradScaler(
            "cuda", 
            enabled=self.cfg.fp16
        )

        self.save = False
        self.writer = None
        
        # ckpt loading priority: conti > pretrained_ckpt
        if conti:
            self.save = conti  # ckpt subfolder name is same as opt.conti
            ckpt = torch.load(os.path.join(self.CKPT_DIR, conti, "ckpt_latest.pt"), 
                              map_location=self.model_device,
                              weights_only=False)
            self.model.actor.load_state_dict(ckpt["weights"])
            self.current_iters = ckpt["current_iters"]
            self.last_ep = ckpt["last_ep"]
        elif self.cfg.pretrained_ckpt:
            ckpt = torch.load(self.cfg.pretrained_ckpt, 
                              map_location=self.model_device,
                              weights_only=False)
            self.model.actor.load_state_dict(ckpt["weights"])
            self.current_iters = 0
            self.last_ep = -1
        else:
            self.current_iters = 0
            self.last_ep = -1

        # if save path is explicitly specified, then overwrite
        if save:
            self.save = save

        decay, no_decay = self.model.parameter_groups()
        self.optimizer = optim.AdamW([
            {"params": decay, "lr": self.cfg.max_lr, "weight_decay": self.cfg.wd},
            {"params": no_decay, "lr": self.cfg.max_lr, "weight_decay": 0.0}
        ])
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.ema = ExponentialMovingAverage(params, decay=self.cfg.ema_decay)

        # model score
        self.best_score = None
        self.larger_better = False

        if conti:
            print("[INFO] resume training from iter: {}".format(ckpt["current_iters"]))
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.best_score = ckpt["best_score"]
            if "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])
            else:
                print("[INFO] Key ema not found in checkpoint, skip loading ema model")
        elif self.cfg.pretrained_ckpt:
            print("[INFO] load pretrained ckpt from iter: {}".format(ckpt["current_iters"]))
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scaler.load_state_dict(ckpt["scaler"])
            if "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])
            else:
                print("[INFO] Key ema not found in checkpoint, skip loading ema model")
        
        self.scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.num_warmup,
            last_epoch=self.last_ep,
        )
        if conti:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        
        self._is_first_save = True

    @classmethod
    def preprocess_data(
        cls, 
        data: Dict[str, Tensor], 
        device,
    ):
        for k in data:
            if isinstance(data[k], Tensor):
                data[k] = data[k].to(device, non_blocking=True)

        return data

    def compute_metrics(self, data: Dict[str, Tensor]):
        data = self.preprocess_data(data, self.model_device)
        total_loss, metrics = self.model(
            obs_rgbs=data["obs_rgbs"], 
            obs_masks=data.get("obs_masks", None),
            obs_norm_xys=data["obs_norm_xys"],
            obs_extrinsics=data["obs_extrinsics"],
            prompt_text=data["prompt_text"],

            current_ee_pose=data["current_ee_pose"],
            history_ee_states=data["history_ee_states"],
            gt_future_ee_states=data["gt_future_ee_states"], 
            valid_ee_mask=data["valid_ee_mask"], 
            inference=False,
            fp16=self.scaler.is_enabled(),
        )

        return total_loss, metrics

    def log_metrics(self, metrics: dict):
        if self.save:
            if self.writer is None:
                log_dir = os.path.join(self.LOG_DIR, self.save)
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
            self.writer.add_scalar(
                "lr", self.scheduler.get_last_lr()[0], self.current_iters)
            for key, val in metrics.items():
                self.writer.add_scalar(key, val, self.current_iters)

    def save_model(self, fname: str, best_score: float, latest_score: float):
        if self.save and self._is_first_save:
            cfg_save_path = os.path.join(self.CKPT_DIR, self.save, "{}.json".format(self.launch_time_str))
            self.cfg.dump(cfg_save_path)
            self._is_first_save = False

        if self.save:
            ckpt_dir = os.path.join(self.CKPT_DIR, self.save)
            os.makedirs(ckpt_dir, exist_ok=True)
            to_save = {
                "weights": self.model.actor.state_dict(),
                "current_iters": self.current_iters,
                "last_ep": self.last_ep, 
                "lr": self.scheduler.get_last_lr()[0], 
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_score": best_score,
                "latest_score": latest_score
            }
            if self.cfg.ema_enabled:
                to_save["ema"] = self.ema.state_dict()
            torch.save(to_save, os.path.join(ckpt_dir, fname))
            print("[INFO] Save to {}".format(os.path.join(ckpt_dir, fname)))

    def fitting(self):
        averages = {}
        self.model.train()
        while self.current_iters <= self.cfg.max_iterations:
            for data in self.train_loader:
                self.current_iters += 1
                self.optimizer.zero_grad()
                loss, metrics = self.compute_metrics(data)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("[INFO] NaN or Inf occured in loss, skip")
                    self.current_iters -= 1
                    continue

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if self.cfg.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                self.scheduler.step()

                if self.current_iters >= self.cfg.ema_start and self.cfg.ema_enabled:
                    self.ema.update()

                print_strings = []
                for key, val in metrics.items():
                    if key not in averages:
                        averages[key] = AverageMeter()
                    averages[key].append(val)
                    print_strings.append("{} = {:.3e}".format(key, averages[key].avg()))

                print("[INFO] {}/{} | {} | lr = {:.3e}".format(
                    self.current_iters, self.cfg.max_iterations, " | ".join(print_strings),
                    self.scheduler.get_last_lr()[0]))

                ### save ckpt and log
                if self.current_iters % self.cfg.save_latest_interval == 0:
                    avg_metrics = {k: v.avg() for k, v in averages.items()}
                    latest_score = avg_metrics["total_loss"]
                    
                    if (
                        (self.best_score is None) or
                        (self.larger_better and (latest_score > self.best_score)) or 
                        (not self.larger_better and (latest_score < self.best_score)) 
                    ):
                        self.best_score = latest_score
                        save_best = True
                    else:
                        save_best = False

                    self.save_model("ckpt_latest.pt", self.best_score, latest_score)
                    if save_best:
                        self.save_model("ckpt_best.pt", self.best_score, latest_score)
                
                if (self.current_iters % self.cfg.save_interval == 0) and (self.cfg.save_interval > 0):
                    avg_metrics = {k: v.avg() for k, v in averages.items()}
                    latest_score = avg_metrics["total_loss"]
                    self.save_model("ckpt_{:0>7d}.pt".format(self.current_iters), 
                                    self.best_score, latest_score)
                
                if self.current_iters % self.cfg.log_interval == 0:
                    avg_metrics = {"train/"+k: v.avg() for k, v in averages.items()}
                    self.log_metrics(avg_metrics)
                    for key in averages.keys():
                        averages[key].reset()
                
                if self.current_iters > self.cfg.max_iterations:
                    break
            
            self.last_ep += 1


if __name__ == "__main__":
    trainer = Trainer()
    trainer.fitting()
