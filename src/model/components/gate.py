import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms import RMSNorm

class GatedMLP(nn.Module):
    """
    Gated MLP (SwiGLU 变体) - 借鉴自 LLM 的高效特征蒸馏模块。
    结构: Down(SiLU(Gate(LN(x))) * Up(LN(x)))
    """
    def __init__(self, input_dim, output_dim, intermediate_size=None, bias=True):
        super().__init__()
        if intermediate_size is None:
            # 默认 8/3 倍扩展，并对齐到 64 的倍数
            intermediate_size = int(output_dim * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            
        self.norm = RMSNorm(input_dim)
        self.gate = nn.Linear(input_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(input_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_dim, bias=bias)
        self.act_func = F.silu

    def forward(self, x):
        # 先进行归一化，防止深层数值爆炸
        x = self.norm(x)
        # Down(SiLU(Gate(x)) * Up(x))
        return self.down_proj(self.act_func(self.gate(x)) * self.up_proj(x))