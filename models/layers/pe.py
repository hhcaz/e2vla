import math
import torch
from einops import rearrange
from torch import nn, Tensor


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, temperature=1e4):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x: Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.temperature) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RoPE(nn.Module):
    def __init__(self, feature_dim, position_dim, num_heads=1, temperature=1e4):
        super().__init__()
        assert feature_dim % (position_dim * num_heads) == 0
        self.head_dim = feature_dim // num_heads
        self.feature_dim = feature_dim
        self.position_dim = position_dim
        self.num_heads = num_heads
        self.temperature = temperature
    
    @staticmethod
    def embed_rotary(x: Tensor, pe: Tensor):
        """
        - x: (batch_size, num_heads, length, head_dim)
        - pe: (batch_size, length, head_dim, 2)
        """
        assert x.shape[-2:] == pe.shape[-3:-1]
        if x.dim() == 4: pe = pe.unsqueeze(1)  # (batch_size, 1, length, head_dim, 2)
        cos, sin = pe.type_as(x).unbind(-1)
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x = x * cos + x2 * sin
        return x
    
    @torch.no_grad()
    def forward(self, x: Tensor):
        """
        Arguments:
        - x: (..., position_dim)

        Returns:
        - pe: (..., head_dim, 2)
        """
        # B, L, PD = x.size()
        PD = x.shape[-1]
        assert PD == self.position_dim

        HD = self.feature_dim // self.num_heads
        div_term = torch.exp(
            torch.arange(0, HD // PD, 2, dtype=torch.float, device=x.device)
            * (-math.log(self.temperature) / (HD // PD))
        )  # (hd//pd/2,)

        x = x.unsqueeze(-1)  # (..., pd, 1)
        sinx = torch.sin(x * div_term)  # (..., pd, hd//pd/2)
        cosx = torch.cos(x * div_term)  # (..., pd, hd//pd/2)

        sinx = sinx.repeat_interleave(2, dim=-1)  # (..., pd, hd//pd)
        cosx = cosx.repeat_interleave(2, dim=-1)  # (..., pd, hd//pd)

        sinx = sinx.flatten(-2)  # (..., pd, hd//pd) -> (..., hd)
        cosx = cosx.flatten(-2)  # (..., pd, hd//pd) -> (..., hd)
        position_code = torch.stack([cosx, sinx], dim=-1)  # (..., hd, 2)

        if position_code.requires_grad:
            position_code = position_code.detach()
        return position_code



class PRoPE(nn.Module):
    @staticmethod
    def embed_q(q: Tensor, cwT: Tensor):
        """
        Args:
            q: (B, NH, Lq, HD)
            cwT: (B, Lq, 4, 4), Gi, world to camera
        """
        # gi^T @ qi -> Q @ gi
        q = rearrange(q, "b h l (d f) -> b h l d f", f=4)
        q = q @ cwT.unsqueeze(1)  # (B, H, Lq, D//4, 4)
        q = rearrange(q, "b h l d f -> b h l (d f)")
        return q
    
    @staticmethod
    def embed_kv(kv: Tensor, wcT: Tensor):
        """
        Args:
            kv: (B, NH, Lk, HD)
            wcT: (B, Lk, 4, 4), Gj^{-1}, camera to world
        """
        # gj^{-1} @ kj -> K @ gj^{-1}^T
        kv = rearrange(kv, "b h l (d f) -> b h l d f", f=4)
        kv = kv @ wcT.unsqueeze(1).transpose(-1, -2)  # (B, H, Lk, D//4, 4)
        kv = rearrange(kv, "b h l d f -> b h l (d f)")
        return kv
    
    @staticmethod
    def embed_out(out: Tensor, cwT: Tensor):
        """
        Args:
            out: (B, NH, Lo, HD)
            cwT: (B, Lq, 4, 4), Gi, world to camera
        """
        # gi @ oi -> O @ gi^T
        out = rearrange(out, "b h l (d f) -> b h l d f", f=4)
        out = out @ cwT.unsqueeze(1).transpose(-1, -2)
        out = rearrange(out, "b h l d f -> b h l (d f)")
        return out

