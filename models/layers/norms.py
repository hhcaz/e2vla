import torch
from torch import nn, Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)
    
    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        # to float32 first for numeric issues
        output = self._norm(x.float()).type_as(x)
        return (output * self.weight) if self.elementwise_affine else output


if __name__ == "__main__":
    to_kwargs = {
        "device": "cuda:0",
        "dtype": torch.bfloat16
    }

    rms_torch = nn.RMSNorm(16).to(**to_kwargs)
    rms_mine = RMSNorm(16).to(**to_kwargs)

    x = torch.randn(16).to(**to_kwargs)
    print(x)

    y1 = rms_torch(x)
    y2 = rms_mine(x)

    delta = y1 - y2
    print(delta.abs().max())
    print(delta.abs().mean())


