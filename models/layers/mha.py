import torch
from einops import rearrange
from torch import nn, Tensor
from enum import IntFlag, auto
import torch.nn.functional as F
from .norms import RMSNorm
from . import pe


class ProjOpt(IntFlag):
    QK = auto()  # use 1 linear layer, to_qk, note v is not projected
    QKV = auto()  # use 1 linear layer: to_qkv
    Q_KV = auto()  # use 2 linear layers: to_q, to_kv
    Q_K_V = auto()  # use 3 linear layers: to_q, to_k, to_v
    DEFAULT = Q_K_V


class MySimpleMHA(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        dropout=0., 
        kdim=None, 
        vdim=None, 
        proj_opt=ProjOpt.DEFAULT, 
        flash=True,
        bias=False,
        qk_norm=False,
        pe_type="rope",
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.dropout = dropout
        self.flash = flash
        self.qk_norm = qk_norm
        self.pe_type = pe_type

        self.proj_opt = proj_opt
        if proj_opt == ProjOpt.QKV:
            self.to_qkv = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        elif proj_opt == ProjOpt.Q_KV:
            self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.to_kv = nn.Linear(self.kdim, 2*embed_dim, bias=bias)
        elif proj_opt == ProjOpt.Q_K_V:
            self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.to_k = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.to_v = nn.Linear(self.vdim, embed_dim, bias=bias)
        elif proj_opt == ProjOpt.QK:
            self.to_qk = nn.Linear(embed_dim, 2*embed_dim, bias=bias)
        
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.to_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        # nn.MultiheadAttention use xavier_uniform initialization
        if self.proj_opt == ProjOpt.QKV:
            nn.init.xavier_uniform_(self.to_qkv.weight)
        elif self.proj_opt == ProjOpt.Q_KV:
            nn.init.xavier_uniform_(self.to_q.weight)
            nn.init.xavier_uniform_(self.to_kv.weight)
        elif self.proj_opt == ProjOpt.Q_K_V:
            nn.init.xavier_uniform_(self.to_q.weight)
            nn.init.xavier_uniform_(self.to_k.weight)
            nn.init.xavier_uniform_(self.to_v.weight)
        elif self.proj_opt == ProjOpt.QK:
            nn.init.xavier_uniform_(self.to_qk.weight)

    @classmethod
    def check_expand_attn_mask(cls, attn_mask: Tensor, B_H_Lx_Lc):
        """
        Arguments:
        - attn_mask: (B, Lc) or (B, Lx, Lc) or (B, H, Lx, Lc)
        - B_H_Lx_Lc: tuple of ints, (B, H, Lx, Lc)

        Returns:
        - attn_mask: (B, H, Lx, Lc)
        """
        if attn_mask is not None:
            B, H, Lx, Lc = B_H_Lx_Lc
            attn_dim = attn_mask.dim()
            assert attn_dim in (2, 3, 4)
            if attn_dim == 2:
                assert attn_mask.shape == (B, Lc), \
                    "attn_mask shape = {}, (B, Lc) = ({}, {})".format(attn_mask.shape, B, Lc)
                attn_mask = attn_mask[:, None, None, :].expand(B, H, Lx, Lc)
            elif attn_dim == 3:
                assert attn_mask.shape == (B, Lx, Lc), \
                    "attn_mask shape = {}, (B, Lx, Lc) = ({}, {}, {})".format(attn_mask.shape, B, Lx, Lc)
                attn_mask = attn_mask[:, None, :, :].expand(B, H, Lx, Lc)
            elif attn_dim == 4:
                assert attn_mask.shape == (B, H, Lx, Lc), \
                    "attn_mask shape = {}, (B, H, Lx, Lc) = ({}, {}, {})".format(attn_mask.shape, B, H, Lx, Lc)
        return attn_mask

    def project_qkv(self, x: Tensor, c: Tensor):
        if self.proj_opt == ProjOpt.QKV:
            assert x is c, "id(x) = {:x}, id(c) = {:x}".format(id(x), id(c))
            qkv = rearrange(self.to_qkv(x), "b l (n h c) -> b h n l c", n=3, h=self.num_heads)
            q, k, v = torch.unbind(qkv, dim=2)
        elif self.proj_opt == ProjOpt.Q_KV:
            q = rearrange(self.to_q(x), "b l (h c) -> b h l c", h=self.num_heads)
            kv = rearrange(self.to_kv(c), "b l (n h c) -> b h n l c", n=2, h=self.num_heads)
            k, v = torch.unbind(kv, dim=2)
        elif self.proj_opt == ProjOpt.Q_K_V:
            q = rearrange(self.to_q(x), "b l (h c) -> b h l c", h=self.num_heads)
            k = rearrange(self.to_k(c), "b l (h c) -> b h l c", h=self.num_heads)
            v = rearrange(self.to_v(c), "b l (h c) -> b h l c", h=self.num_heads)
        elif self.proj_opt == ProjOpt.QK:
            qk = rearrange(self.to_qk(x), "b l (n h c) -> b h n l c", n=2, h=self.num_heads)
            q, k = torch.unbind(qk, dim=2)
            v = rearrange(c, "b l (h c) -> b h l c", h=self.num_heads)
        
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        return q, k, v

    def forward(
        self, 
        x: Tensor, 
        c: Tensor, 
        x_pe: Tensor = None, 
        c_pe: Tensor = None, 
        attn_mask: Tensor = None,
        return_attn_weights = False,
        average_attn_weights = True,
    ):
        """
        Arguments:
        - x: (B, Lx, C)
        - c: (B, Lc, C), context
        - x_pe:
            * if pe_type == rope, shape of x_pe = (B, Lx, H, 2)
            * if pe_type == prope: shape of x_pe = (B, Lx, 4, 4), camera to world
        - c_pe:
            * if pe_type == rope: shape of c_pe = (B, Lc, H, 2)
            * if pe_type == prope: shape of c_pe = (B, Lc, 4, 4), camera to world
        - attn_mask: (B, Lc) or (B, Lx, Lc) or (B, NH, Lx, Lc)
            * if boolean: then attn_bias at attn_mask will be -inf
            * if float/double: then attn_bias will be attn_mask
        - return_attn_weights: bool
        - average_attn_weights: bool

        Returns:
        - out: (B, Lx, C)
        - (Optional) attn_weight: (B, Lx, Lc) if average_attn_weights 
                             else (B, H, Lx, Lc)
        """
        use_sdpa = self.flash and (not return_attn_weights)

        if use_sdpa:
            return self.forward_sdpa(x, c, x_pe, c_pe, attn_mask)
        else:
            return self.forward_eager(x, c, x_pe, c_pe, attn_mask, return_attn_weights, average_attn_weights)
    
    @classmethod
    def _attn_impl_eager(
        cls,
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        training: bool, 
        attn_mask: Tensor = None, 
        drop_attn: float = 0.0, 
        return_attn_weights = False,
        average_attn_weights = True,
    ):
        B, H, Lq, D = q.shape
        B, H, Lk, _ = k.shape
        
        attn_mask = cls.check_expand_attn_mask(attn_mask, (B, H, Lq, Lk))
        corr: Tensor = (q @ k.transpose(-1, -2)) / (D ** 0.5)  # (B, H, Lx, Lc)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                corr.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                corr = corr + attn_mask
        corr = corr - corr.amax(dim=-1, keepdim=True).detach()  # (B, H, Lx, Lc)
        attn_weight = torch.softmax(corr, dim=-1)
        attn_weight = attn_weight.masked_fill(attn_weight.isnan(), 0.)  # nan occurs if one row is fully masked out
        attn_weight = torch.dropout(attn_weight, drop_attn, training)
        out = attn_weight @ v  # (B, H, Lx, HD)

        if return_attn_weights and average_attn_weights:
            if H == 1:
                attn_weight = attn_weight.squeeze(1)
            else:
                attn_weight = attn_weight.mean(1)  # (B, Lx, Lc)
        
        if return_attn_weights:
            return (out, attn_weight), attn_mask
        else:
            return out, attn_mask
    
    @classmethod
    def _attn_impl_sdpa(
        cls,
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        training: bool, 
        attn_mask: Tensor = None, 
        drop_attn: float = 0.0, 
        **kwargs
    ):
        B, H, Lq, D = q.shape
        B, H, Lk, _ = k.shape
        
        attn_mask = cls.check_expand_attn_mask(attn_mask, (B, H, Lq, Lk))
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = attn_mask,
            dropout_p = drop_attn if training else 0., 
            is_causal = False
        )
        out = out.masked_fill(out.isnan(), 0.)  # nan occurs if one row is fully masked out
        return out, attn_mask

    def embed_qkv(
        self, 
        q: Tensor, k: Tensor, v: Tensor, 
        x_pe: Tensor, c_pe: Tensor
    ):
        # x_pe can be x_wcT
        # c_pe can be c_wcT
        x_pe_inv = None
        if self.pe_type == "rope":
            if x_pe is not None:
                q = pe.RoPE.embed_rotary(q, x_pe)
            if c_pe is not None:
                k = pe.RoPE.embed_rotary(k, c_pe)
        elif self.pe_type == "prope":
            if x_pe is not None:
                x_cwT = x_pe_inv = x_pe.inverse()
                q = pe.PRoPE.embed_q(q, x_cwT)
            if c_pe is not None:
                k = pe.PRoPE.embed_kv(k, c_pe)
                v = pe.PRoPE.embed_kv(v, c_pe)
        else:
            raise TypeError("Unknown pe_type: {}".format(self.pe_type))
        return q, k, v, x_pe_inv
    
    def embed_o(
        self, o: Tensor, x_pe_inv: Tensor
    ):
        if self.pe_type == "prope":
            if x_pe_inv is not None:
                o = pe.PRoPE.embed_out(o, x_pe_inv)
        return o

    def forward_eager(
        self, 
        x: Tensor, 
        c: Tensor, 
        x_pe: Tensor = None, 
        c_pe: Tensor = None, 
        attn_mask: Tensor = None,
        return_attn_weights = False,
        average_attn_weights = True,
    ):
        q, k, v = self.project_qkv(x, c)
        q, k, v, x_pe_inv = self.embed_qkv(q, k, v, x_pe, c_pe)
        
        ret = self._attn_impl_eager(
            q, k, v, 
            training=self.training, 
            attn_mask=attn_mask, 
            drop_attn=self.dropout, 
            return_attn_weights=return_attn_weights, 
            average_attn_weights=average_attn_weights
        )

        if return_attn_weights:
            (out, attn_weight), attn_mask = ret
        else:
            out, attn_mask = ret
        
        out = self.embed_o(out, x_pe_inv)
        out = rearrange(out, "b h l c -> b l (h c)", h=self.num_heads)
        out = self.to_out(out)

        if return_attn_weights:
            return (out, attn_weight), attn_mask
        else:
            return out, attn_mask
    
    def forward_sdpa(
        self, 
        x: Tensor, 
        c: Tensor, 
        x_pe: Tensor = None, 
        c_pe: Tensor = None, 
        attn_mask: Tensor = None
    ):
        q, k, v = self.project_qkv(x, c)
        q, k, v, x_pe_inv = self.embed_qkv(q, k, v, x_pe, c_pe)
        
        out, attn_mask = self._attn_impl_sdpa(
            q, k, v,
            training=self.training,
            attn_mask=attn_mask,
            drop_attn=self.dropout,
        )

        out = self.embed_o(out, x_pe_inv)
        out = rearrange(out, "b h l c -> b l (h c)", h=self.num_heads)
        out = self.to_out(out)

        return out, attn_mask


