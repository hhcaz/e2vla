import os
import json
from typing import List, Dict
from dataclasses import dataclass, field, asdict

from data_utils import datasets
from data_utils.dataset_base import H5DatasetMapBase


@dataclass
class TrainConfig(object):

    model: str = "base"  # choices are ["tiny", "small", "base"]
    pretrained_ckpt: str | None = None  # ckpt path of pretrained model

    bs: int = 32  # batch size
    workers: int = 4  # num_workers
    fp16: bool = True  # enable mixed precision training (fp32 and bfloat16)

    grad_clip: float = -1  # <= 0 disables the grad clip
    max_lr: float = 1e-4  # maximum learning rate
    wd: float = 1e-2  # weight decay
    num_warmup: int = int(10e3)  # warm up steps

    ema_enabled: bool = False
    ema_start: int = int(400e3)
    ema_decay: float = 0.9995

    dataset_classes: List[type[H5DatasetMapBase] | str] = field(default_factory=list)
    dataset_weights: List[float] | None = None  # len = len(datasets)
    sample_multiplex: int = 1   # set this to a large number (e.g. 1000) if the total number of samples are small

    log_interval: int = 100
    save_interval: int = int(100e3)  # ckpt are named as ckpt_{iter}.pt, set <0 to disable this
    save_latest_interval: int = 2000  # ckpt are named as ckpt_latest.pt
    max_iterations: int = int(600e3)
    
    def __post_init__(self):
        for i, D in enumerate(self.dataset_classes):
            if isinstance(D, str):
                self.dataset_classes[i] = getattr(datasets, D)
            else:
                assert issubclass(D, H5DatasetMapBase)
    
    def dump(self, path: str):
        items = asdict(self)
        dataset_classes = items["dataset_classes"]
        for i, D in enumerate(dataset_classes):
            if issubclass(D, H5DatasetMapBase):
                dataset_classes[i] = D.__name__
            else:
                assert isinstance(D, str)
        
        save_folder = os.path.dirname(path)
        os.makedirs(save_folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(items, fp, ensure_ascii=False, indent=4)
    
    @classmethod
    def load(cls, path: str):
        with open(path, "r") as fp:
            items = json.load(fp)
        return cls(**items)


CONFIGS: Dict[str, TrainConfig] = {}
CONFIGS["debug"] = TrainConfig()
CONFIGS["pretrain"] = TrainConfig(
    dataset_classes=[
        datasets.Droid,
        datasets.Maniskill,
        datasets.MetaWorld,
    ],
    dataset_weights=[1, 1, 10]
)
CONFIGS["finetune_libero_spatial"] = TrainConfig(
    dataset_classes=[datasets.LiberoSpatial],
    dataset_weights=[1],
    sample_multiplex=1000
)
CONFIGS["finetune_libero_object"] = TrainConfig(
    dataset_classes=[datasets.LiberoObject],
    dataset_weights=[1],
    sample_multiplex=1000
)
CONFIGS["finetune_libero_goal"] = TrainConfig(
    dataset_classes=[datasets.LiberoGoal],
    dataset_weights=[1],
    sample_multiplex=1000
)
CONFIGS["finetune_libero_10"] = TrainConfig(
    dataset_classes=[datasets.Libero10],
    dataset_weights=[1],
    sample_multiplex=1000
)
