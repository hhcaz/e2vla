import torch
from typing import Iterable
from torch import nn, Tensor
from .mha import MySimpleMHA, ProjOpt


class AdaLN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x: Tensor, t: Tensor):
        """
        Arguments:
        - x: (B, L, C)
        - t: (B, C) or (B, L, C)
        """
        assert x.dim() == 3
        assert t.dim() in (2, 3)
        if t.dim() == 2:
            t = t.unsqueeze(1)
        assert t.shape[1] == 1 or t.shape[1] == x.shape[1]
        scale, shift = torch.chunk(self.modulation(t), 2, dim=-1)
        x = x * (1 + scale) + shift
        return x


class NormOrAdaLN(nn.Module):
    def __init__(self, hdim, use_adaln: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(hdim, elementwise_affine=not use_adaln)
        self.use_adaln = use_adaln
        if self.use_adaln:
            self.adaln = AdaLN(hdim)
    
    def forward(self, x: Tensor, film: Tensor = None):
        x = self.norm(x)
        if self.use_adaln:
            x = self.adaln(x, film)
        return x


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, use_adaln=False):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        # norm or adaLN
        self.norm = NormOrAdaLN(embed_dim, use_adaln)
        # deepnorm related
        self.alpha = 1.0
    
    def residual_path(self, x: Tensor, residual: Tensor):
        if self.alpha == 1:
            return x + residual
        else:
            return x + self.alpha * residual

    def forward(self, x: Tensor, film: Tensor = None):
        residual = x
        x = self.ff(x)
        output = self.residual_path(x, residual)
        output = self.norm(output, film)
        return output


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_adaln=False, 
                 proj_opt=ProjOpt.Q_KV, bias=False, qk_norm=False, pe_type="rope"):
        super().__init__()
        self.attn = MySimpleMHA(
            embed_dim, num_heads, dropout=dropout, proj_opt=proj_opt,
            bias=bias, qk_norm=qk_norm, pe_type=pe_type
        )
        self.dropout = nn.Dropout(dropout)
        # norm or adaLN
        self.norm = NormOrAdaLN(embed_dim, use_adaln)
        # deepnorm related
        self.alpha = 1.0

    def residual_path(self, x: Tensor, residual: Tensor):
        if self.alpha == 1:
            return x + residual
        else:
            return x + self.alpha * residual

    def forward(
        self, 
        query, 
        value, 
        query_pe=None, 
        value_pe=None, 
        value_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        residual = query
        attn_output, value_mask = self.attn(
            x=query,
            c=value,
            x_pe=query_pe,
            c_pe=value_pe,
            attn_mask=value_mask,
            return_attn_weights=return_attn_weights,
            average_attn_weights=average_attn_weights
        )
        if return_attn_weights:
            attn_output, attn_weight = attn_output
        attn_output = self.dropout(attn_output)
        output = self.residual_path(attn_output, residual)
        output = self.norm(output, film)
        
        return ((output, attn_weight) if return_attn_weights else output), value_mask


class SelfAttentionLayer(CrossAttentionLayer):
    def __init__(self, embed_dim, num_heads, dropout=0, use_adaln=False, 
                 proj_opt=ProjOpt.QKV, bias=False, qk_norm=False, pe_type="rope"):
        super().__init__(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            use_adaln=use_adaln, 
            proj_opt=proj_opt, 
            bias=bias, 
            qk_norm=qk_norm, 
            pe_type=pe_type
        )

    def forward(
        self, 
        query, 
        query_pe=None, 
        query_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        residual = query
        attn_output, query_mask = self.attn(
            x=query,
            c=query,
            x_pe=query_pe,
            c_pe=query_pe,
            attn_mask=query_mask,
            return_attn_weights=return_attn_weights,
            average_attn_weights=average_attn_weights
        )
        if return_attn_weights:
            attn_output, attn_weight = attn_output
        attn_output = self.dropout(attn_output)
        output = self.residual_path(attn_output, residual)
        output = self.norm(output, film)
        
        return ((output, attn_weight) if return_attn_weights else output), query_mask


class FFWCrossAttentionLayers(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.0, use_adaln=False,
                 proj_opt=ProjOpt.Q_KV, bias=False, qk_norm=False, pe_type="rope"):
        super().__init__()
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_layers.append(
                CrossAttentionLayer(
                    embed_dim=embed_dim, 
                    num_heads=num_heads, 
                    dropout=dropout, 
                    use_adaln=use_adaln, 
                    proj_opt=proj_opt, 
                    bias=bias, 
                    qk_norm=qk_norm,
                    pe_type=pe_type
                )
            )
            self.ffn_layers.append(
                FFN(
                    embed_dim=embed_dim, 
                    hidden_dim=4*embed_dim, 
                    dropout=dropout, 
                    use_adaln=use_adaln
                )
            )
    
    def forward(
        self, 
        query, 
        value, 
        query_pe=None, 
        value_pe=None, 
        value_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        """
        - query, value: (B, Lq or Lv, C)
        - query_pe, value_pe: (B, Lq or Lv, HD, 2) or (B, Lq or Lv, 4, 4)
        - value_mask: (B, Lv)
        - film: (B, C)
        """
        outputs = []
        for i in range(self.num_layers):
            query, value_mask = self.attn_layers[i](
                query=query, 
                value=value, 
                query_pe=query_pe, 
                value_pe=value_pe, 
                value_mask=value_mask, 
                film=film,
                return_attn_weights=return_attn_weights, 
                average_attn_weights=average_attn_weights
            )
            if return_attn_weights:
                query, attn_weight = query
            query = self.ffn_layers[i](x=query, film=film)
            outputs.append((query, attn_weight) if return_attn_weights else query)
        return outputs


class FFWSelfAttentionLayers(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0, use_adaln=False, 
                 proj_opt=ProjOpt.QKV, bias=False, qk_norm=False, pe_type="rope"):
        super().__init__()
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_layers.append(
                SelfAttentionLayer(
                    embed_dim=embed_dim, 
                    num_heads=num_heads, 
                    dropout=dropout, 
                    use_adaln=use_adaln, 
                    proj_opt=proj_opt, 
                    bias=bias, 
                    qk_norm=qk_norm,
                    pe_type=pe_type
                )
            )
            self.ffn_layers.append(
                FFN(
                    embed_dim=embed_dim, 
                    hidden_dim=4*embed_dim, 
                    dropout=dropout, 
                    use_adaln=use_adaln
                )
            )

    def forward(
        self, 
        query, 
        query_pe=None, 
        query_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        """
        - query, value: (B, Lq or Lv, C)
        - query_pe, value_pe: (B, Lq or Lv, C, 2)
        - value_mask: (B, Lv)
        - film: (B, C)
        """
        outputs = []
        for i in range(self.num_layers):
            query, query_mask = self.attn_layers[i](
                query=query, 
                query_pe=query_pe, 
                query_mask=query_mask, 
                film=film,
                return_attn_weights=return_attn_weights, 
                average_attn_weights=average_attn_weights
            )
            if return_attn_weights:
                query, attn_weight = query
            query = self.ffn_layers[i](x=query, film=film)
            outputs.append((query, attn_weight) if return_attn_weights else query)
        return outputs


############################################################################
# Implementation of DeepNorm: https://arxiv.org/abs/2203.00555

@torch.no_grad()
def set_alpha_beta(alpha: float, beta: float, modules: Iterable[nn.Module]):
    for m in modules:
        if isinstance(m, (FFN, SelfAttentionLayer, CrossAttentionLayer)):
            m.alpha = alpha
        
        if isinstance(m, FFN):
            for l in m.ff.modules():
                if isinstance(l, nn.Linear):
                    l.weight.mul_(beta)
        
        if isinstance(m, MySimpleMHA):
            m.to_out.weight.mul_(beta)
            
            if m.proj_opt == ProjOpt.QKV:
                w = m.to_qkv.weight
                out_dim, in_dim = w.shape
                v_start = out_dim // 3 * 2
                w[v_start:].mul_(beta)
            
            if m.proj_opt == ProjOpt.Q_KV:
                w = m.to_kv.weight
                out_dim, in_dim = w.shape
                v_start = out_dim // 2
                w[v_start:].mul_(beta)
            
            if m.proj_opt == ProjOpt.Q_K_V:
                w = m.to_v.weight
                w.mul_(beta)


def init_xncoder(N: int, xncoder: nn.Module):
    alpha = (2*N) ** 0.25
    beta = (8*N) ** (-0.25)
    print("[INFO] DeepNorm, N = {}, alpha = {:.3f}, beta = {:.3f}".format(N, alpha, beta))
    set_alpha_beta(alpha, beta, xncoder.modules())


def init_encoder_decoder(N: int, M: int, encoder: nn.Module, decoder: nn.Module):
    alpha_enc = 0.81 * (N**4 * M) ** (1.0/16)
    beta_enc = 0.87 * (N**4 * M) ** (-1.0/16)
    set_alpha_beta(alpha_enc, beta_enc, encoder.modules())
    
    alpha_dec = (3*M) ** 0.25
    beta_dec = (12*M) ** -0.25
    set_alpha_beta(alpha_dec, beta_dec, decoder.modules())

