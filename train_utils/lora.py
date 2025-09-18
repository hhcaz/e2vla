import math
import torch
from typing import List
from torch import nn, Tensor
from models.layers.mha import MySimpleMHA


class LoraLinear(nn.Module):
    def __init__(self, lin: nn.Linear, r: int):
        super().__init__()
        self.r = r
        self.lin = lin

        for p in self.lin.parameters():
            p.requires_grad_(False)

        self.A = nn.Parameter(torch.zeros(lin.in_features, r).to(lin.weight.device))
        self.B = nn.Parameter(torch.zeros(r, lin.out_features).to(lin.weight.device))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, math.sqrt(5))
        with torch.no_grad():
            scaling = 1.0 / self.r
            self.A.mul_(scaling)

    def forward(self, x: Tensor):
        return self.lin(x) + (x @ self.A) @ self.B


def replace_with_lora_linear(model: nn.Module, r: int):
    num_replaced = 0
    for name, m in model.named_modules():
        if isinstance(m, MySimpleMHA):
            for key in ["to_q", "to_k", "to_v", "to_qk", "to_kv", "to_qkv"]:
                if hasattr(m, key):
                    num_replaced += 1
                    print("[INFO] [{}] Replace {}".format(num_replaced, name))
                    n = len(key.replace("to_", ""))
                    lora_linear = LoraLinear(getattr(m, key), r * n)
                    setattr(m, key, lora_linear)
    return model, num_replaced


def get_lora_parameters(model: nn.Module):
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoraLinear):
            params.append(m.A)
            params.append(m.B)
    return params


@torch.no_grad()
def merge_lora_linear(model: nn.Module, inplace: bool = True):

    index = 0

    def _merge_lora_linear_inplace(model: nn.Module, target_module_name=""):
        nonlocal index
        for name, m in model.named_children():
            if isinstance(m, LoraLinear):
                index += 1
                linear = m.lin
                lora_weight = torch.einsum("i r, r o -> o i", m.A, m.B)

                if inplace:
                    linear.weight.copy_(linear.weight + lora_weight)
                    setattr(model, name, linear)
                else:
                    new_linear = nn.Linear(
                        in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=linear.bias is not None,
                        device=linear.weight.device,
                        dtype=linear.weight.dtype,
                    )
                    new_linear.weight.copy_(linear.weight + lora_weight)
                    if new_linear.bias is not None:
                        new_linear.bias.copy_(linear.bias)
                    setattr(model, name, new_linear)

                print("[INFO] [{}] Merge {}".format(index, target_module_name + "." + name))
            else:
                _merge_lora_linear_inplace(m, target_module_name + "." + name if target_module_name else name)
        return model
    
    merged = _merge_lora_linear_inplace(model, "")
    for m in merged.modules():
        assert not isinstance(m, LoraLinear)
    
    return merged


