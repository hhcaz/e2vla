import cv2
import torch
import numpy as np
from torch import Tensor
from einops import rearrange
from typing import Dict, List, Tuple
from data_utils.dataset_base import draw_ee_proj


def visualize_traj(
    data: Dict[str, Tensor], 
    future_ee_states: List[Tensor],
    colors: List[Tuple[int, int, int]]
):
    """draw current ee pose and 

    Args:
        data (Dict[str, Tensor]): remove the batch dimension
        future_ee_states (List[Tensor]): you can plot multiple trajs
        colors (List[Tuple[int, int, int]]): each traj can has different color

    Returns:
        bgrs: image to visualize
    """
    # data["obs_rgbs"]: (To, ncam, C, H, W)
    To, ncam, _, H, W = data["obs_rgbs"].shape
    rgb = rearrange(data["obs_rgbs"][-1], "n c h w -> n h w c")  # latest time
    
    # data["K"]: (ncam, 3, 3)
    K = data["K"]  # (ncam, 3, 3)
    K_np = K.cpu().numpy()
    
    # data["obs_extrinsics"]: (To, ncam, 4, 4)
    wcT = data["obs_extrinsics"][-1]  # (ncam, 4, 4)
    wcT_np = wcT.cpu().numpy()
    
    valid_ee_mask = data["valid_ee_mask"]  # (nee)
    valid_ee_mask_np = valid_ee_mask.cpu().numpy()
    
    nhist, nee, _ = data["history_ee_states"].shape
    history_weTs = data["history_ee_states"][:, :, :16].view(nhist, nee, 4, 4)
    history_weTs_np = history_weTs.cpu().numpy()
    
    bgrs = np.ascontiguousarray(rgb.flip(-1).cpu().numpy())  # (ncam, H, W, C)
    
    for typei, ee_states in enumerate(future_ee_states):
        # future_ee_states: (Ta, nee, 4*4+1)
        Ta, nee, _ = ee_states.shape
        weTs = ee_states[:, :, :16].view(Ta, nee, 4, 4)  # (Ta, nee, 4, 4)

        ceTs = (
            rearrange(torch.inverse(wcT), "ncam r c -> () () ncam r c") @ 
            rearrange(weTs, "Ta nee r c -> Ta nee () r c")
        )  # (Ta, nee, ncam, 4, 4)
        cets = ceTs[..., :3, 3]  # (Ta, nee, ncam, 3)

        proj_norm = cets[..., :2] / cets[..., 2:3]  # (Ta, nee, ncam, 2)
        fxy = K[..., [0, 1], [0, 1]]; cxy = K[..., [0, 1], [2, 2]]  # (ncam, 2)
        proj_pix = proj_norm * fxy + cxy  # (Ta, nee, ncam, 2)
        proj_pix_np: np.ndarray = proj_pix.cpu().numpy()  # (Ta, nee, ncam, 2)
    
        for eei in range(nee):
            if not valid_ee_mask_np[eei]:
                continue
            
            for cami in range(ncam):
                for x, y in proj_pix_np[:, eei, cami]:
                    cv2.circle(bgrs[cami], (int(x), int(y)), 
                               radius=2, color=colors[typei], thickness=-1)

                if typei == 0:
                    # draw current ee poses only once
                    draw_ee_proj(
                        bgrs[cami], 
                        K=K_np[cami], 
                        cwT=np.linalg.inv(wcT_np[cami]), 
                        pose=history_weTs_np[-1, eei]
                    )

    bgrs = rearrange(bgrs, "n h w c -> h (n w) c")
    bgrs = np.ascontiguousarray(bgrs)
    return bgrs


