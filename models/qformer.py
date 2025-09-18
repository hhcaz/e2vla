import torch
from typing import Optional
from torch import nn, Tensor
from .layers.utils import concat_mask
from .layers.attn_dn import SelfAttentionLayer, CrossAttentionLayer, FFN, init_xncoder


class QBlockITM(nn.Module):
    def __init__(self, hdim: int, num_heads: int):
        super().__init__()
        self.self_attn_text = SelfAttentionLayer(hdim, num_heads, bias=True, qk_norm=True)
        self.cross_attn_vision = CrossAttentionLayer(hdim, num_heads, bias=True, qk_norm=True)
        self.ffn_query = FFN(hdim, 4*hdim)
        self.ffn_text = FFN(hdim, 4*hdim)
    
    def forward(
        self, 
        query: Tensor, 
        x_vision: Tensor, 
        mask_vision: Optional[Tensor], 
        x_text: Tensor,
        mask_qt: Optional[Tensor],
    ):
        """
        Args:
            query (Tensor): (B, Lq, C)
            x_vision (Tensor): (B, Lv, C)
            mask_vision (Optional[Tensor]): (B, Lv) or None
            x_text (Tensor): (B, Lt, C)
            mask_qt (Optional[Tensor]): (B, Lq+Lt) or None
        
        Returns
        -------
            query (Tensor): (B, Lq, C)
            x_text (Tensor): (B, Lt, C)
            mask_vision (Optional[Tensor]): (B, Lv) or None
            mask_qt (Optional[Tensor]): (B, Lq+Lt) or None
        """
        qt = torch.cat([query, x_text], dim=1)
        qt, mask_qt = self.self_attn_text(
            query=qt, 
            query_mask=mask_qt
        )
        
        query, x_text = torch.split(qt, [query.shape[1], x_text.shape[1]], dim=1)
        x_text = self.ffn_text(x_text)

        query, mask_vision = self.cross_attn_vision(
            query=query, 
            value=x_vision, 
            value_mask=mask_vision
        )
        query = self.ffn_query(query)
        return query, x_text, mask_vision, mask_qt


class QFormerITM(nn.Module):
    def __init__(self, hdim: int, num_heads: int, num_layers: int, num_queries: int):
        super().__init__()
        self.num_layers = num_layers
        self.queries = nn.Parameter(torch.randn(1, num_queries, hdim))
        self.layers = nn.ModuleList([QBlockITM(hdim, num_heads) 
                                     for _ in range(num_layers)])
        self.reset_parameters()
    
    def reset_parameters(self):
        ### init params
        nn.init.trunc_normal_(self.queries, std=0.02)
        init_xncoder(self.num_layers*2, self.layers)

    def forward(
        self, 
        x_vision: Tensor, 
        mask_vision: Optional[Tensor], 
        x_text: Tensor,
        mask_text: Optional[Tensor]
    ):
        """
        Args:
            x_vision (Tensor): (B, Lv, C)
            mask_vision (Optional[Tensor]): (B, Lv) or None
            x_text (Tensor): (B, Lt, C)
            mask_text (Optional[Tensor]): (B, Lt) or None
        
        Returns:
            query (Tensor): (B, Lq, C)
            x_text (Tensor): (B, Lt, C)
            mask_qt (Optional[Tensor]): (B, Lq+Lt) or None
        """
        B = x_vision.shape[0]
        query = self.queries.expand(B, -1, -1)
        mask_qt = concat_mask(mask0=None, mask1=mask_text,
                              L0=query.shape[1], L1=x_text.shape[1])
        
        for layer in self.layers:
            query, x_text, mask_vision, mask_qt = layer(
                query=query,
                x_vision=x_vision,
                mask_vision=mask_vision,
                x_text=x_text,
                mask_qt=mask_qt
            )
        
        return query, x_text, mask_qt
