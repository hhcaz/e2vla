import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, Tensor
from diffusers import DDIMScheduler
from typing import Optional, Tuple, Dict, List

from .dit import DiT
from .qformer import QFormerITM
from .layers.utils import simple_mlp
from .layers.utils import concat_mask
from .layers.pe import SinusoidalPosEmb
from .layers.attn_dn import FFWSelfAttentionLayers, init_xncoder
from .layers.rot_transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


class ContextEncoder(nn.Module):
    def __init__(self, hdim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.proj_v = nn.ModuleList([
            simple_mlp([768, hdim*4, hdim], ln=True),  # for dinov2 vision embeds
            simple_mlp([768, hdim*4, hdim], ln=True),  # for siglip vision embeds
        ])
        self.proj_l = nn.ModuleList([
            simple_mlp([768, hdim*4, hdim], ln=True)  # for siglip language embeds
        ])
        self.proj_pe = simple_mlp([2, hdim, hdim], ln=True)  # for normalized coordinates

        self.main_cam_embed = nn.Parameter(torch.zeros(hdim))
        self.pre_attn = DiT(hdim, num_heads, num_layers//2, use_adaln=False, pe_type="prope")
        self.qformer = QFormerITM(hdim, num_heads, num_layers=1, num_queries=64)
        self.post_attn = FFWSelfAttentionLayers(hdim, num_heads, num_layers//2, use_adaln=False,
                                                bias=True, qk_norm=True)
        self.reset_parameters()

    def reset_parameters(self):
        init_xncoder(self.post_attn.num_layers, self.post_attn)
        nn.init.zeros_(self.proj_pe[-1].weight)

    def forward(
        self,
        vl_obs: Dict[str, Tensor],
        vl_feature: Dict[str, Tensor], 
        fp16: bool
    ):
        """
        Args:
            vl_obs (Dict[str, Tensor]):
                - rgb: (B, To, ncam, 3, H, W)
                - mask: (B, To, ncam, H, W)
                - norm_xy: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
                - text: List (length=B) of prompt
                - extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            
            vl_feature (Dict[str, Tensor]):
                - norm_xy_ds: (B, Ncam, Lv, 2)
                - vision_embeds: List (length=num_layer) of (B, Ncam, Lv, C)
                - vision_mask: (B, Ncam, Lv) or None
                - lang_embeds: List (length=num_layer) of (B, La, C)
                - lang_mask: (B, La)
                - extrinsics: (B, Ncam, 4, 4)

            fp16: if True, use bfloat16
        
        Returns
        -------
            context: (B, Ncam*Lt, hdim)
            context_mask: (B, Ncam*Lt)
        """
        B, T, Ncam, _, H, W = vl_obs["rgb"].shape
        obs_extrinsics = vl_obs["extrinsics"]  # (B, To, Ncam, 4, 4)

        # get relative camera extrinsic referring to camera 0 at the latest timestep
        cam0_extr_ref = torch.inverse(obs_extrinsics[:, -1:, 0:1]) @ obs_extrinsics  # (B, To, Ncam, 4, 4)
        x_v: Tensor = self.proj_v[0](vl_feature["vision_embeds"][0]) + \
                      self.proj_v[1](vl_feature["vision_embeds"][1])  # (B, Ncam, Lv, C)
        x_l: Tensor = self.proj_l[0](vl_feature["lang_embeds"][0])    # (B, Ncam, La, C)
        norm_xy_ds = vl_feature["norm_xy_ds"]  # (B, Ncam, Lv, 2)
        mask_v = vl_feature["vision_mask"]  # (B, Ncam, Lv)
        mask_l = vl_feature["lang_mask"]  # (B, La)

        # camera pose as positional encoding
        extrinsic_pe = cam0_extr_ref[:, -1]  # (B, Ncam, 4, 4), select the latest frame
        extrinsic_pe = extrinsic_pe[:, :, None, :, :].expand(B, Ncam, x_v.shape[-2], 4, 4)
        extrinsic_pe = rearrange(extrinsic_pe, "b n l r c -> b (n l) r c")

        # get the latest frame and projection
        x_v = rearrange(x_v, "b n l c -> b (n l) c")
        pe2d = rearrange(self.proj_pe(norm_xy_ds), "b n l c -> b (n l) c")
        if mask_v is not None:
            mask_v = rearrange(mask_v, "b n l -> b (n l)")

        # SA before qformer
        with torch.autocast(
            x_v.device.type, 
            torch.bfloat16 if fp16 else torch.float32
        ):
            x_v: Tensor = self.pre_attn(
                x=x_v + pe2d,  # add positional encoding
                x_pe=extrinsic_pe, 
                x_mask=mask_v,
                conds=[x_l],
                cond_masks=[mask_l],
                films=None
            )

        x_v = x_v.clone()
        L = x_v.shape[1] // Ncam
        x_v[:, :L] = x_v[:, :L] + self.main_cam_embed  # the first camera as main camera

        with torch.autocast(
            x_v.device.type, 
            torch.bfloat16 if fp16 else torch.float32
        ):
            query, x_l, _ = self.qformer(
                x_vision=x_v,
                mask_vision=mask_v,
                x_text=x_l,
                mask_text=mask_l,
            )

            query = self.post_attn(
                query=query,
            )[-1]
        
        cond = torch.cat([query, x_l], dim=1)
        cond_mask = concat_mask(mask0=None, mask1=mask_l,
                                L0=query.shape[1], L1=x_l.shape[1])
        return cond, cond_mask


class DiffusionHead(nn.Module):
    """
    - Input: image observations and noisy action at `t`
    - Output: noisy action at `t-1`
    """

    def __init__(self, hdim: int, num_heads: int, act_dim: int, num_layers: int):
        super().__init__()
        self.act_dim = act_dim
        self.num_layers = num_layers
        self.hist_enc = simple_mlp([act_dim-1, hdim, hdim], ln=True)
        self.traj_enc = simple_mlp([act_dim, hdim, hdim], ln=True)
        self.abs_pos_enc = simple_mlp([3, hdim, hdim], ln=True)
        self.traj_time_embed = SinusoidalPosEmb(hdim)
        self.denoising_time_embed = nn.Sequential(
            SinusoidalPosEmb(hdim),
            simple_mlp([hdim, hdim, hdim], ln=True)
        )

        ### traj self attn + traj-context cross attn
        self.traj_context_attn = DiT(hdim, num_heads, num_layers, use_adaln=True)

        ### final mlp
        self.act_head = simple_mlp([hdim, hdim, act_dim], ln=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.abs_pos_enc[-1].weight)
        nn.init.zeros_(self.act_head[-1].weight)
    
    def pos_rel2abs(self, cur_wcT: Tensor, cur_weT: Tensor, t3r6: Tensor):
        """
        Args:
            cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
            cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
            t3r6 (Tensor): (B, T, 9)
        
        Returns:
            traj_cet (Tensor), traj ee pos in camera frame, shape (B, T, 3)
        """
        ecT = torch.inverse(cur_weT) @ cur_wcT  # (B, 4, 4)
        ecR = ecT[:, :3, :3]  # (B, 3, 3)
        
        e1e2R = ecR[:, None] @ rotation_6d_to_matrix(t3r6[..., 3:]) @ ecR[:, None].transpose(-1, -2)
        e1e2t = (ecR[:, None] @ t3r6[..., :3].unsqueeze(-1)).squeeze(-1)
        
        e1e2T = e1e2t.new_zeros(*e1e2t.shape[:-1], 4, 4)
        e1e2T[..., :3, :3] = e1e2R
        e1e2T[..., :3, 3] = e1e2t
        e1e2T[..., 3, 3] = 1

        traj_ceT = (torch.inverse(cur_wcT) @ cur_weT)[:, None] @ e1e2T
        traj_cet = traj_ceT[..., :3, 3]  # (B, T, 3)
        return traj_cet

    def forward(
        self, 
        denoise_timestep: Tensor, 
        trajectory: Tensor, 
        cur_wcT: Tensor, 
        cur_weT: Tensor, 
        history: Tensor, 
        conds: List[Tensor],
        cond_masks: Optional[List[Optional[Tensor]]],
        fp16: bool
    ):
        """
        Args:
            denoise_timestep: (B,), denoising time step
            trajectory: (B, Ta, act_dim)
            cur_wcT: (B, 4, 4)
            cur_weT: (B, 4, 4)
            history: (B, nhist, act_dim)
            conds: [(B, Lc, hdim)]
            cond_masks: [(B, Lc)]
            films: [(B, hdim)]
            fp16 (bool): use bfloat16

        Returns:
            action_epsilon: (B, Ta, act_dim)
        """
        denoise_time_embed = self.denoising_time_embed(denoise_timestep)  # (B, hdim)
        film = denoise_time_embed

        # noisy trajectory add temporal positional embeddings
        B, nhist, _ = history.shape
        B, Ta, _ = trajectory.shape
        hist_feats = self.hist_enc(history[:, :, :self.act_dim-1])  # (B, nhist, hdim)
        traj_feats = self.traj_enc(trajectory[:, :, :self.act_dim])  # (B, Ta, hdim)

        full_traj_feats = torch.cat([hist_feats, traj_feats], dim=1)  # (B, nhist+Ta, hdim)
        full_traj_time_pe = self.traj_time_embed(torch.arange(nhist + Ta).to(traj_feats))
        full_traj_feats = full_traj_feats + full_traj_time_pe[None].expand(B, -1, -1)  # (B, nhist+Ta, hdim)

        # get absolute pos under camera, add additional positional encoding
        with torch.no_grad():
            full_traj_t3r6 = torch.cat([history[..., :9], trajectory[..., :9]], dim=1)  # (B, nhist+Ta, 9)
            abs_pos = self.pos_rel2abs(cur_wcT, cur_weT, full_traj_t3r6)
        full_traj_feats = full_traj_feats + self.abs_pos_enc(abs_pos)

        with torch.autocast(
            denoise_time_embed.device.type, 
            torch.bfloat16 if fp16 else torch.float32
        ):
            full_traj_feats = self.traj_context_attn(
                x=full_traj_feats, 
                x_pe=None, 
                x_mask=None,
                conds=conds,
                cond_masks=cond_masks, 
                films=[film]*len(conds)
            )

        traj_feats = full_traj_feats[:, nhist:nhist+Ta]  # (B, Ta, hdim)
        action = self.act_head(traj_feats)
        return action


class ActionExpert(nn.Module):
    def __init__(
        self, 
        hdim: int, 
        num_heads: int, 
        num_context_layers: int,
        num_diffusion_layers: int, 
        diffusion_timesteps: int = 100, 
        inference_timesteps: Optional[int] = None, 
    ):
        super().__init__()
        self.context_encoder = ContextEncoder(hdim, num_heads, num_layers=num_context_layers)
        self.dp_head = DiffusionHead(hdim, num_heads, self.act_dim, num_layers=num_diffusion_layers)

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False
        )

        self.diffusion_timesteps = diffusion_timesteps
        if inference_timesteps is None:
            inference_timesteps = max(diffusion_timesteps//5, 10)
        self.inference_timesteps = inference_timesteps
        self.inference_scheduler = self.noise_scheduler

    @property
    def act_dim(self):
        """dimension of action defined in camera frame"""
        return 10

    def iterative_denoise(
        self, 
        traj_shape: Tuple[int, int, int],
        fixed_inputs: Dict[str, Tensor],
        initial_noise: Optional[Tensor] = None
    ):
        """
        Args:
            trajectory_shape: (B, Ta, act_dim)
            fixed_inputs: inputs for diffusion head
            initial_noise: (B, Ta, act_dim) or None

        Returns:
            trajectory: (B, Ta, act_dim)
        """
        if initial_noise is None:
            B, Ta, _ = traj_shape
            device = next(iter(fixed_inputs.values())).device
            initial_noise = torch.randn(B, Ta, self.act_dim, device=device)
        
        self.inference_scheduler.set_timesteps(self.inference_timesteps)
        trajectory = initial_noise
        for t in self.inference_scheduler.timesteps:
            out = self.dp_head(
                t * torch.ones(trajectory.shape[0], device=trajectory.device), 
                trajectory,
                **fixed_inputs
            )
            trajectory = self.inference_scheduler.step(
                out[..., :self.act_dim], t, trajectory[..., :self.act_dim]
            ).prev_sample
        return trajectory

    def forward(
        self, 
        vl_obs: Dict[str, Tensor],
        vl_feature: Dict[str, Tensor], 
        current_ee_pose: Tensor, 
        history_ee_states: Tensor, 
        gt_future_ee_states: Tensor, 
        valid_ee_mask: Tensor, 
        inference: bool, 
        fp16: bool,
    ):
        """
        Args:
            vl_obs (Dict[str, Tensor]):
                - rgb: (B, To, ncam, 3, H, W)
                - mask: (B, To, ncam, H, W)
                - norm_xy: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
                - text: List (length=B) of prompt
                - extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            
            vl_feature (Dict[str, Tensor]):
                - norm_xy_ds: (B, Ncam, Lv, 2)
                - vision_embeds: List (length=num_layer) of (B, Ncam, Lv, C)
                - vision_mask: (B, Ncam, Lv) or None
                - lang_embeds: List (length=num_layer) of (B, La, C)
                - lang_mask: (B, La)
                - extrinsics: (B, Ncam, 4, 4)

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
        latest_cam_poses = vl_obs["extrinsics"][:, -1]  # (B, Ncam, 4, 4)
        current_cam_pose = latest_cam_poses[:, 0]  # first camera, (B, 4, 4)
        
        # patch features as current observation context in diffusion
        cond, cond_mask = self.context_encoder(
            vl_obs=vl_obs,
            vl_feature=vl_feature,
            fp16=fp16,
        )
        
        valid_ee_per_batch = valid_ee_mask.sum(dim=-1)  # (B,)
        sel_index = torch.cat([torch.empty(n, dtype=torch.long).fill_(b) 
                               for b, n in enumerate(valid_ee_per_batch.tolist())]
                              ).to(valid_ee_mask.device)
        B_expand = len(sel_index)  # B'
        
        history_action = states2action(
            current_cam_pose[sel_index], 
            current_ee_pose[valid_ee_mask], 
            history_ee_states.transpose(1, 2)[valid_ee_mask]
        )  # (B', nhist, 10)

        B, Ta, Nee, _ = gt_future_ee_states.shape
        if not inference:
            gt_future_action = states2action(
                current_cam_pose[sel_index], 
                current_ee_pose[valid_ee_mask], 
                gt_future_ee_states.transpose(1, 2)[valid_ee_mask]
            )  # (B', Ta, 10)

        fixed_inputs = dict(
            history=history_action,  # history in camera 0, shape (B', nhist, act_dim)
            conds=[cond[sel_index]],
            cond_masks=[cond_mask[sel_index] if cond_mask is not None else cond_mask],
            cur_wcT=current_cam_pose[sel_index],      # (B', 4, 4) 
            cur_weT=current_ee_pose[valid_ee_mask],   # (B', 4, 4)
            fp16=fp16
        )

        ###################### Inference ######################
        if inference:
            pred_actions = self.iterative_denoise(
                traj_shape=(B_expand, Ta, self.act_dim),
                fixed_inputs=fixed_inputs
            )  # (B', Ta, act_dim)
            pred_future_ee_states = action2states(
                current_cam_pose[sel_index],    # (B', 4, 4)
                current_ee_pose[valid_ee_mask], # (B', 4, 4)
                pred_actions  # (B', Ta, act_dim)
            )  # (B', Ta, 4*4+1)
            
            # put the predition back
            pred_future_ee_states_full = pred_future_ee_states.new_zeros(B, Nee, Ta, 4*4+1)
            pred_future_ee_states_full[..., :16] = torch.eye(4).ravel().to(pred_future_ee_states)
            pred_future_ee_states_full[valid_ee_mask] = pred_future_ee_states
            return pred_future_ee_states_full.transpose(1, 2).contiguous()  # (B, Ta, Nee, 17)

        ###################### Training ######################
        # sample noise
        noise = torch.randn(B_expand, Ta, self.act_dim, 
                            device=gt_future_ee_states.device)

        # sample a random timestep
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            size=(B_expand,), 
            device=noise.device
        )

        # add noise to the clean trajectory
        noisy_trajectory = self.noise_scheduler.add_noise(
            gt_future_action, noise,
            timesteps
        )

        # one step denoising
        pred = self.dp_head(timesteps, noisy_trajectory, **fixed_inputs)
        target = get_target(gt_future_action, noise, timesteps, self.noise_scheduler)
        
        # # drop too aggresive actions
        # debug_gt_ee_pose = gt_future_ee_states.transpose(1, 2)[valid_ee_mask][..., :16].reshape(B_expand, -1, 4, 4)
        # debug_gt_ee_pos = debug_gt_ee_pose[..., :3, 3]  # (B', Ta, 3)
        # delta_norm = (debug_gt_ee_pos[:, 1:] - debug_gt_ee_pos[:, :-1]).norm(dim=-1)
        # debug_mask = delta_norm < 0.2  # (B', Ta-1)
        # debug_mask[..., 1:-1] = debug_mask[..., 1:-1] & debug_mask[..., :-2] & debug_mask[..., 2:]
        # debug_mask = torch.cat([debug_mask[:, 0:1], debug_mask], dim=-1)  # (B', Ta)
        # # filter 
        # pred = pred[debug_mask]
        # target = target[debug_mask]

        # loss calculation
        pos_loss = F.l1_loss(pred[..., 0:3], target[..., 0:3], reduction="mean")
        rot_loss = F.l1_loss(pred[..., 3:9], target[..., 3:9], reduction="mean")
        openness_loss = F.l1_loss(pred[..., 9:10], target[..., 9:10], reduction="mean")

        total_loss = 30 * pos_loss + 10 * rot_loss + 10 * openness_loss
        metrics = {
            "pos_loss": pos_loss.item(),
            "rot_loss": rot_loss.item(),
            "openness_loss": openness_loss.item(),
            "total_loss": total_loss.item()
        }
        return total_loss, metrics


def space_ee2cam(cur_wcT: Tensor, cur_weT: Tensor, fut_weT: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        fut_weT (Tensor): (B, T, 4, 4), future ee pose in world frame
    
    Returns:
        t3r6 (Tensor): some repr of ^{cam} v _{ee} * dt, shape (B, T, 9)
    """
    e1e2T = torch.inverse(cur_weT[:, None]) @ fut_weT  # (B, T, 4, 4)
    e1e2R = e1e2T[:, :, :3, :3]  # (B, T, 3, 3)
    e1e2t = e1e2T[:, :, :3, 3]  # (B, T, 3)

    ceT = torch.inverse(cur_wcT) @ cur_weT  # (B, 4, 4)
    ceR = ceT[:, :3, :3]  # (B, 3, 3)
    
    r = matrix_to_rotation_6d(ceR[:, None] @ e1e2R @ ceR[:, None].transpose(-1, -2))
    t = (ceR[:, None] @ e1e2t.unsqueeze(-1)).squeeze(-1)
    t3r6 = torch.cat([t, r], dim=-1)
    return t3r6


def space_cam2ee(cur_wcT: Tensor, cur_weT: Tensor, t3r6: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        t3r6 (Tensor): (B, T, 9)
    
    Returns:
        fut_weT (Tensor), future ee pose in world frame, shape (B, T, 4, 4)
    """
    ecT = torch.inverse(cur_weT) @ cur_wcT  # (B, 4, 4)
    ecR = ecT[:, :3, :3]  # (B, 3, 3)
    
    e1e2R = ecR[:, None] @ rotation_6d_to_matrix(t3r6[..., 3:]) @ ecR[:, None].transpose(-1, -2)
    e1e2t = (ecR[:, None] @ t3r6[..., :3].unsqueeze(-1)).squeeze(-1)
    
    e1e2T = e1e2t.new_zeros(*e1e2t.shape[:-1], 4, 4)
    e1e2T[..., :3, :3] = e1e2R
    e1e2T[..., :3, 3] = e1e2t
    e1e2T[..., 3, 3] = 1

    fut_weT = cur_weT[:, None] @ e1e2T
    return fut_weT


def states2action(cur_wcT: Tensor, cur_weT: Tensor, ee_states: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        ee_states (Tensor): (B, T, 16 or 17)
    
    Returns:
        action (Tensor): (B, T, 9 or 10)
    """
    B, Ta, C = ee_states.shape
    weT = ee_states[:, :, :16].view(B, Ta, 4, 4)
    t3r6 = space_ee2cam(cur_wcT, cur_weT, weT)
    
    if C == 16:
        return t3r6
    else:
        openness = (ee_states[:, :, -1:] - 0.5) * 2  # normalize gripper openness
        return torch.cat([t3r6, openness], dim=-1)


def action2states(cur_wcT: Tensor, cur_weT: Tensor, action: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        action (Tensor): (B, T, 9 or 10)
    
    Returns:
        ee_states (Tensor): (B, T, 16 or 17)
    """
    B, Ta, C = action.shape
    t3r6 = action[:, :, :9]
    weT = space_cam2ee(cur_wcT, cur_weT, t3r6).view(B, Ta, 16)
    
    if C == 9:
        return weT
    else:
        openness = action[:, :, -1:] / 2 + 0.5  # denormalize gripper openness
        return torch.cat([weT, openness], dim=-1)


def get_target(traj: Tensor, noise: Tensor, timesteps: Tensor, scheduler: DDIMScheduler):
    """returns supervision depending on scheduler type"""
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        target = noise
    if pred_type == "sample":
        target = traj
    if pred_type == "v_prediction":
        target = scheduler.get_velocity(traj, noise, timesteps) 
    return target


def count_parameters():
    model = ActionExpert(
        hdim=256,
        diffusion_timesteps=100,
        num_heads=4,
    )

    modules = [
        model
    ]

    num_param = 0
    for m in modules:
        for p in m.parameters():
            if not p.requires_grad:
                continue
            
            num_param += p.numel()

    print("[INFO] Total {:.3f}M trainable parameters"
          .format(num_param / 1e6))


if __name__ == "__main__":
    count_parameters()

