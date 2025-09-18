from typing import Optional
from torch import nn, Tensor

from .vlm import VLM
from .action_expert import ActionExpert


class VLA(nn.Module):
    def __init__(
        self, 
        hdim: int, 
        num_heads: int,
        num_actor_context_layers: int,
        num_actor_diffusion_layers: int, 

        diffusion_timesteps: int = 100, 
        inference_timesteps: int = 20, 
    ):
        super().__init__()
        self.vlm = VLM()
        self.actor = ActionExpert(
            hdim=hdim,
            num_heads=num_heads,
            num_context_layers=num_actor_context_layers,
            num_diffusion_layers=num_actor_diffusion_layers, 
            diffusion_timesteps=diffusion_timesteps,
            inference_timesteps=inference_timesteps,
        )
    
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if m.bias is not None and m.bias.requires_grad:
                    # Do not modify the bias in fronzen backbones!!!
                    nn.init.zeros_(m.bias)
    
    def parameter_groups(self):
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if (
                name.endswith(".bias") or 
                "norm" in name.lower() or 
                "qformer.queries" in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        
        return decay, no_decay
    
    def forward(
        self, 
        obs_rgbs: Tensor,
        obs_masks: Optional[Tensor], 
        obs_norm_xys: Tensor, 
        obs_extrinsics: Tensor, 
        prompt_text: Optional[Tensor], 

        current_ee_pose: Tensor, 
        history_ee_states: Tensor, 
        gt_future_ee_states: Tensor, 
        valid_ee_mask: Tensor, 
        inference: bool, 
        fp16: bool,
    ):
        """
        Args:
            obs_rgbs: (B, To, ncam, 3, H, W)
            obs_masks: (B, To, ncam, H, W)
            obs_norm_xys: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
            obs_extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            prompt_text: (B, Lang, E) or None, language instruction

            current_ee_pose: (B, Nee, 4, 4), ^{world}_{ee} T
            history_ee_states: (B, nhist, Nee, 4*4+1), in world frame,
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
            gt_future_ee_states: (B, Ta, Nee, 4*4+1), ground truth future actions, in world frame
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
                * Note: if `inference` is True, we only derive prediction actions shape from gt_future_ee_states
            valid_ee_mask: (B, Nee), only compute loss on these end-effectors
            inference: if True, returns the predicted trajectory, otherwise returns loss and metrics for logging
            fp16: if True, use bfloat16
        
        Returns
        -------
        (if inference is True)
            pred_future_ee_states (Tensor): (B, Ta, Nee, 4*4+1)
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
        (else)
            loss (Tensor): scalar tensor
            metrics (Dict[str, Tensor]): metrics for logging
        """
        vl_obs, vl_feature = self.vlm(
            obs_rgbs=obs_rgbs,
            obs_masks=obs_masks,
            obs_norm_xys=obs_norm_xys,
            obs_extrinsics=obs_extrinsics,

            prompt_text=prompt_text,
            fp16=fp16
        )

        # ### select the latest frame for higher execution frequency of action expert
        # obs_rgbs = obs_rgbs[:, -1:]             # (B, To=1, ncam, 3, H, W)
        # if obs_masks is not None:
        #     obs_masks = obs_masks[:, -1:]       # (B, To=1, ncam, H, W)
        # obs_norm_xys = obs_norm_xys[:, -1:]     # (B, To=1, ncam, 2, H, W)
        # obs_extrinsics = obs_extrinsics[:, -1:] # (B, To=1, ncam, 4, 4)

        return self.actor(
            vl_obs=vl_obs,
            vl_feature=vl_feature,

            current_ee_pose=current_ee_pose,
            history_ee_states=history_ee_states,
            gt_future_ee_states=gt_future_ee_states,
            valid_ee_mask=valid_ee_mask, 
            inference=inference,
            fp16=fp16
        )


def vla_tiny(
    diffusion_timesteps: int = 100, 
    inference_timesteps: int = 20, 
):
    return VLA(
        hdim=192,
        num_heads=3,
        num_actor_context_layers=8,
        num_actor_diffusion_layers=4,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps
    )


def vla_small(
    diffusion_timesteps: int = 100, 
    inference_timesteps: int = 20, 
):
    return VLA(
        hdim=384,
        num_heads=6,
        num_actor_context_layers=8,
        num_actor_diffusion_layers=4,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps
    )


def vla_base(
    diffusion_timesteps: int = 100, 
    inference_timesteps: int = 20, 
):
    return VLA(
        hdim=768,
        num_heads=12,
        num_actor_context_layers=8,
        num_actor_diffusion_layers=4,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps
    )



def count_parameters():
    # model = vla_tiny()
    # model = vla_small()
    model = vla_base()

    modules = [
        model
    ]

    num_total = 0
    num_trainable = 0
    for m in modules:
        for p in m.parameters():
            num_total += p.numel()
            if p.requires_grad:
                num_trainable += p.numel()

    print("[INFO] Total {:.3f}M parameters, {:.3f}M frozen, {:.3f}M trainable"
          .format(num_total / 1e6, (num_total - num_trainable) / 1e6, num_trainable / 1e6))


if __name__ == "__main__":
    count_parameters()
