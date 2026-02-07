import torch
import numpy as np
from typing import Literal

@torch.jit.script
def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """将输入张量归一化到 [-1, 1] 范围。
    
    Args:
        x: 输入张量 (N, dims)。
        lower: 最小值 (N, dims) 或 (dims,)。
        upper: 最大值 (N, dims) 或 (dims,)。
    """
    offset = (lower + upper) * 0.5
    return 2 * (x - offset) / (upper - lower)

@torch.jit.script
def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """将 [-1, 1] 范围的张量反向映射回 (lower, upper)。
    """
    offset = (lower + upper) * 0.5
    return x * (upper - lower) * 0.5 + offset

@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """将输入张量截断到 (lower, upper) 范围。
    """
    return torch.max(torch.min(x, upper), lower)

@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """将输入张量归一化为单位长度。
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def convert_quat(quat: torch.Tensor | np.ndarray, to: Literal["xyzw", "wxyz"] = "xyzw") -> torch.Tensor | np.ndarray:
    """在 (w, x, y, z) 和 (x, y, z, w) 四元数格式间转换。
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"四元数形状错误: {quat.shape}")
        
    if isinstance(quat, np.ndarray):
        return np.roll(quat, -1 if to == "xyzw" else 1, axis=-1)
    else:
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=torch.float)
        return quat.roll(-1 if to == "xyzw" else 1, dims=-1)
