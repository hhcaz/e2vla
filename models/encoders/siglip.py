import os
import torch
import torch.nn.functional as F
from typing import List
from einops import rearrange
from torch import nn, Tensor
from torchvision.transforms import v2
from transformers import SiglipProcessor, SiglipTokenizer, SiglipImageProcessor, SiglipModel


class SiglipEncoder(nn.Module):
    REPO_ID = "google/siglip-base-patch16-256"

    def __init__(self):
        super().__init__()

        run_fp16 = torch.cuda.is_bf16_supported()
        self.siglip = SiglipModel.from_pretrained(
            self.REPO_ID,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16 if run_fp16 else torch.float32,
        )

        self.processor = SiglipProcessor.from_pretrained(
            self.REPO_ID,
            use_fast=True
        )

        image_processor: SiglipImageProcessor = self.processor.image_processor
        self.tokenizer: SiglipTokenizer = self.processor.tokenizer
        self.image_tform = v2.Compose([
            v2.Resize(
                size=(image_processor.size["height"], image_processor.size["width"]),
                interpolation=v2.InterpolationMode.BICUBIC
            ),
            v2.Normalize(
                mean=image_processor.image_mean, 
                std=image_processor.image_std
            ),
        ])
    
    @property
    def input_size(self):
        image_processor: SiglipImageProcessor = self.processor.image_processor
        return (
            image_processor.size["height"], 
            image_processor.size["width"]
        )  # for siglip-base-patch16-256, it is (256, 256)
    
    @property
    def output_size(self):
        return (16, 16)
    
    @property
    def patch_size(self):
        return self.siglip.config.vision_config.patch_size  # 16
    
    def forward_siglip_head(self, x_vision: Tensor):
        B, L, C = x_vision.shape

        head = self.siglip.vision_model.head
        prob = head.probe.repeat(B, L, 1).view(B*L, 1, C)  # (B, L, C)
        x_vision = x_vision.view(B*L, 1, C)

        x = head.attention(prob, x_vision, x_vision)[0]  # (B*L, 1, C)
        x = x + head.mlp(head.layernorm(x))
        x = x[:, 0].view(B, L, C)
        return x

    def encode_images(self, rgb: Tensor):
        pixel_values = self.image_tform(rgb)
        vision_outputs = self.siglip.vision_model(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            interpolate_pos_encoding=False,
        )
        x_vision: Tensor = vision_outputs["last_hidden_state"]  # (B, L, C)
        x_vision = self.forward_siglip_head(x_vision)
        x_vision = x_vision / torch.norm(x_vision, dim=-1, keepdim=True)

        gx_vision: Tensor = vision_outputs["pooler_output"]     # (B, C)
        gx_vision = gx_vision / torch.norm(gx_vision, dim=-1, keepdim=True)
        return x_vision, gx_vision

    def encode_texts(self, texts: List[str]):
        input_ids: Tensor = self.tokenizer(
            texts, 
            padding="max_length", 
            return_tensors="pt",
            truncation=True
        )["input_ids"]
        input_ids = input_ids.to(self.siglip.text_model.embeddings.token_embedding.weight.device)
        text_outputs = self.siglip.text_model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        x_text: Tensor = text_outputs["last_hidden_state"]  # (B, L, C), L=64
        x_text = self.siglip.text_model.head(x_text)
        x_text = x_text / torch.norm(x_text, dim=-1, keepdim=True)
        gx_text: Tensor = text_outputs["pooler_output"]     # (B, C)
        gx_text = gx_text / torch.norm(gx_text, dim=-1, keepdim=True)
        return x_text, gx_text

    def match(self, gx_vision: Tensor, gx_text: Tensor):
        logits_per_text = (gx_text @ gx_vision.transpose(-1, -2) * self.siglip.logit_scale.exp() + 
                           self.siglip.logit_bias)
        logits_per_image = logits_per_text.transpose(-1, -2)
        return logits_per_text, logits_per_image


class FrozenEncoder(nn.Module):
    def __init__(self, pool_ks: int = 1):
        super().__init__()
        self.siglip = SiglipEncoder()
        self.resize2 = v2.Resize(self.siglip.input_size, v2.InterpolationMode.BILINEAR)
        self.resize0 = v2.Resize(self.siglip.input_size, v2.InterpolationMode.NEAREST)

        self.pool_ks = pool_ks
        self.pool_aux = nn.AvgPool2d(self.siglip.patch_size * pool_ks)
        self.pool_token = nn.Identity() if pool_ks == 1 else nn.AvgPool2d(pool_ks)
    
    def encode_images(self, rgb: Tensor):
        x_siglip, gx_siglip = self.siglip.encode_images(rgb)
        x_siglip = x_siglip.float()
        gx_siglip = gx_siglip.float()

        if self.pool_ks > 1:
            h, w = self.siglip.output_size
            x_siglip = rearrange(x_siglip, "b (h w) c -> b c h w", h=h, w=w)
            x_siglip = self.pool_token(x_siglip)
            x_siglip = rearrange(x_siglip, "b c h w -> b (h w) c")
        return x_siglip, gx_siglip
    
    def encode_texts(self, texts: List[str]):
        x_text, gx_text = self.siglip.encode_texts(texts)
        return x_text.float(), gx_text.float()
    
    def encode_aux(self, a: Tensor):
        B, C, H, W = a.shape
        assert H == W
        
        bool_type = a.dtype == torch.bool
        float_type = a.is_floating_point()
        int_type = (not bool_type) and (not float_type)
        
        if float_type:
            a_resize = self.resize2(a)
        else:
            a_resize = self.resize0(a.float())
        
        a_ds: Tensor = self.pool_aux(a_resize)
        if bool_type:
            a_ds = a_ds > 1e-4
        elif int_type:
            a_ds = a_ds.to(a.dtype)
        a_ds = rearrange(a_ds, "b c h w -> b (h w) c")
        return a_ds


class Encoder(nn.Module):
    def __init__(self, pool_ks: int = 1):
        super().__init__()
        self.frozen = FrozenEncoder(pool_ks)
        for p in self.frozen.parameters():
            p.requires_grad_(False)
    
    @property
    def output_dim(self):
        return 768

    def encode_mv_images(self, rgb: Tensor):
        """
        Args:
            rgb (Tensor): (B, T, N, 3, H, W)

        Returns:
            x_ds (Tensor): (B, T, N, L, C)
            gx (Tensor): (B, T, N, C)
        """
        B, T, N, C, H, W = rgb.shape
        rgb = rearrange(rgb, "b t n c h w -> (b t n) c h w")
        with torch.no_grad():
            x_ds, gx = self.frozen.encode_images(rgb)
        x_ds = rearrange(x_ds, "(b t n) l c -> b t n l c", b=B, t=T, n=N)
        gx = rearrange(gx, "(b t n) c -> b t n c", b=B, t=T, n=N)
        return x_ds, gx
    
    def pool_mv_masks(self, mask: Tensor):
        if mask is not None:
            B, T, N, H, W = mask.shape
            mask = rearrange(mask, "b t n h w -> (b t n) () h w")
            mask_ds = self.frozen.encode_aux(mask)
            mask_ds = rearrange(mask_ds, "(b t n) l 1 -> b t n l", b=B, t=T, n=N)
        else:
            mask_ds = None
        return mask_ds
    
    def pool_mv_aux(self, aux: Tensor):
        if aux is not None:
            B, T, N, C, H, W = aux.shape
            aux = rearrange(aux, "b t n c h w -> (b t n) c h w")
            aux_ds = self.frozen.encode_aux(aux)
            aux_ds = rearrange(aux_ds, "(b t n) l c -> b t n l c", b=B, t=T, n=N)
        else:
            aux_ds = None
        return aux_ds
    
    def ray_encoding(self, norm_xy: Tensor, extrinsic: Tensor):
        norm_xy_ds = self.pool_mv_aux(norm_xy)
        pe = plucker_ray_pe(norm_xy_ds, extrinsic)
        return norm_xy_ds, pe
    
    def encode_text(self, texts: List[str]):
        with torch.no_grad():
            x, gx = self.frozen.encode_texts(texts)
        return x, gx

    def forward(
        self, 
        rgb: Tensor, 
        mask: Tensor, 
        norm_xy: Tensor, 
        extrinsic: Tensor, 
        **aux_tensors: Tensor
    ):
        """
        Args:
            rgb (Tensor): (B, T, N, 3, H, W)
            mask (Tensor): (B, T, N, H, W)
            norm_xy (Tensor): (B, T, N, 2, H, W)
            extrinsic (Tensor): (B, T, N, 4, 4), ^ref_cam T
            aux_tensors (Tensor): each of (B, T, N, C, H, W)

        Returns:
            obs (Dict[str, Tensor]): image observations
                - x:    tensor of shape (B, To, Ncam, L, C) patch feature
                - gx:   tensor of shape (B, To, Ncam, C), projected global feature, not masked
                - pe:   tensor of shape (B, To, Ncam, L, 6), ray pe
                - mask: tensor of shape (B, To, Ncam, L) or None, patch mask
                - aux:  {aux_user_key: tensor of shape (B, To, Ncam, L, ...)}
        """
        x_ds, gx = self.encode_mv_images(rgb)
        mask_ds = self.pool_mv_masks(mask)
        pe = self.ray_encoding(norm_xy, extrinsic)
        aux_ds = {k:self.pool_mv_aux(a) for k, a in aux_tensors.items()}

        return {
            "x": x_ds,       # (B, T, N, L, C)
            "gx": gx,        # (B, T, N, C)
            "pe": pe,        # (B, T, N, L, 6)
            "mask": mask_ds, # (B, T, N, L) or None
            "aux": aux_ds,   # (B, T, N, L, ...) of each entry
        }


def plucker_ray_pe(norm_xy: Tensor, extrinsic: Tensor):
    """
    Args:
        norm_xy (Tensor): (..., L, 2)
        extrinsic (Tensor): (..., 4, 4), ^w_c T
    
    Returns:
        pe (Tensor): (..., L, 6)
    """
    homo_dir = F.pad(norm_xy, pad=(0, 1), mode="constant", value=1.0)
    direction = F.normalize(homo_dir, dim=-1)  # (..., L, 3)
    rotmat = extrinsic[..., :3, :3]  # (..., 3, 3)
    direction = direction @ rotmat.transpose(-1, -2)  # (..., L, 3), to world rays
    # (..., 3) -> (..., 1, 3) -> (..., L, 3)
    origin = extrinsic[..., :3, 3].unsqueeze(-2).expand_as(direction)  # (..., L, 3)
    pe = torch.cat([direction, torch.cross(direction, origin, dim=-1)], dim=-1)
    return pe


if __name__ == "__main__":
    from IPython import embed
    
    siglip = SiglipEncoder().cuda().eval()
    texts = ["a dog", "an apple a day"]
    
    with torch.no_grad():
        out = siglip.tokenizer(
            texts, 
            padding="max_length", 
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True
        )
        # out = siglip.processor(text=texts, padding="max_length", return_tensors="pt")
    
    print(out)
    embed()

