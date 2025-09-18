import torch
from typing import List
from torch import nn, Tensor
from .encoders.dino import Encoder as Dinov2
from .encoders.siglip import Encoder as Siglip


class VLM(nn.Module):
    def __init__(
        self, 
    ):
        super().__init__()
        
        self.siglip = Siglip()
        self.dinov2 = Dinov2()
        for p in self.siglip.parameters():
            p.requires_grad_(False)
        for p in self.dinov2.parameters():
            p.requires_grad_(False)

    def forward(
        self, 
        obs_rgbs: Tensor,
        obs_masks: Tensor, 
        obs_norm_xys: Tensor, 
        obs_extrinsics: Tensor, 
        prompt_text: List[str], 
        fp16: bool
    ):
        
        with torch.no_grad():
            x_dinov2, gx_dinov2 = self.dinov2.encode_mv_images(obs_rgbs)
            x_siglip, gx_siglip = self.siglip.encode_mv_images(obs_rgbs)
            x_text, gx_text = self.siglip.encode_text(prompt_text)
        
        mask_ds = self.siglip.pool_mv_masks(obs_masks)
        norm_xy_ds = self.siglip.pool_mv_aux(obs_norm_xys)
        
        obs = {
            "rgb": obs_rgbs,
            "mask": obs_masks,
            "norm_xy": obs_norm_xys,
            "extrinsics": obs_extrinsics, 
            "text": prompt_text,
        }
        
        feature = {
            "norm_xy_ds": norm_xy_ds[:, -1],        # (B, Ncam, Lv, 2) 
            "vision_embeds": [x_dinov2[:, -1], 
                              x_siglip[:, -1]],     # List of (B, Ncam, Lv, C)
            "vision_mask": mask_ds[:, -1],          # (B, Ncam, Lv)
            "lang_embeds": [x_text],                # List of (B, La, C)
            "lang_mask": None,                      # (B, La)
            "extrinsics": obs_extrinsics[:, -1]
        }
        # NOTE: xxx[:, -1] selects the latest image observation. We don't use history images

        return obs, feature

