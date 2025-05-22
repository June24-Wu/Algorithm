import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # RMSNorm = x / sqrt(mean(x**2) + eps) * weight
        return self.weight * x / (x.pow(2).mean(dim=-1,keepdim=True) + self.eps).sqrt()


# 测试维度
dim = 512
eps = 1000
x = torch.randn(3, 10, dim, dtype=torch.float32)

# 初始化两个 RMSNorm
ref_weight = torch.randn(dim)

# PyTorch built-in RMSNorm
rms_builtin = nn.RMSNorm(normalized_shape=dim, eps=eps)
rms_builtin.weight.data = ref_weight.clone()

# Custom RMSNorm
rms_custom = RMSNorm(normalized_shape=dim, eps=eps)
rms_custom.weight.data = ref_weight.clone()

# Compute outputs
out_builtin = rms_builtin(x)
out_custom = rms_custom(x)

# Compare
print("Max abs diff:", (out_builtin - out_custom).abs().max().item())
print("All close:", torch.allclose(out_builtin, out_custom, atol=1e-6))