import psutil
import socket
import argparse
from .planner import TrajPlanner
from ..dataset import AlohaPickPlace0903
from shm_transport import expose, run_simple_server, setup_log_level


def get_ipv4_addresses():
    ipv4_list = []
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:  # 只取 IPv4
                ipv4_list.append(addr.address)
    return ipv4_list


def check_ipv4(substr: str):
    if substr in ["localhost", "127.0.0.1", "0.0.0.0"]:
        return substr
    
    ip_list = get_ipv4_addresses()
    for ip in ip_list:
        if ip.startswith(substr):
            if substr != ip:
                print("[INFO] Complete ip from {} to {}".format(substr, ip))
            return ip
    print("[INFO] substr {} not found in all the listed ips: {}"
          .format(substr, ip_list))
    print("[INFO] Fallback to localhost.")
    return "localhost"


import cv2
import torch
import numpy as np

class Service(TrajPlanner):
    @expose()
    def get_config(self):
        return vars(self.config)
    
    @expose()
    def set_config(self, config):
        super().set_config(config)
    
    @expose()
    def reset(self):
        super().reset()
    
    @expose()
    def set_prompt(self, prompt_text: str):
        super().set_prompt(prompt_text)
    
    @expose()
    def add_obs_frame(self, obs_frame):
        super().add_obs_frame(obs_frame)
    
    @expose()
    def get_action(self, draw_traj: bool = False):
        return super().get_action(draw_traj)
    
    @expose()
    def get_action_torch(self, draw_traj: bool = False, compress: bool = True):
        if self.ensemble > 0:
            torch.manual_seed(0)
        ret = super().get_action(draw_traj)
        if ret is None:
            return ret
        
        if ret[-1] is not None and compress:
            img = ret[-1]
            if img.dtype == np.float32:
                img = (img * 255.).clip(0, 255).astype(np.uint8)
            img = cv2.imencode(".jpg", img)[1]
            ret = ret[:-1] + (img,)
        return [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in ret]


def run_service():
    import logging

    setup_log_level(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ema", action="store_true", default=False)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--uri", type=str, default="control_caz")
    parser.add_argument("--ns_host", type=str, default="localhost", help="naming server host")
    parser.add_argument("--ns_port", type=int, default=9090, help="naming server port")
    parser.add_argument("--host", type=str, default="localhost", help="daemon host")
    parser.add_argument("--port", type=int, default=0, help="daemon port")
    parser.add_argument("--ensemble", type=int, default=4, help="ensemble traj for smoothness")
    opt = parser.parse_args()

    assert len(opt.ckpt), "Please specify a valid ckpt path."

    service = Service(
        ckpt_path=opt.ckpt, 
        config=AlohaPickPlace0903.config,
        device="cuda:0",
        ensemble=opt.ensemble,
        use_ema=opt.ema
    )

    run_simple_server(
        service,
        uri_name=opt.uri,
        daemon_host=check_ipv4(opt.host),
        daemon_port=opt.port,
        ns_host=opt.ns_host, 
        ns_port=opt.ns_port,
        multiplex=False
    )


if __name__ == "__main__":
    run_service()


