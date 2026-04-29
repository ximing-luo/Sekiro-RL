from typing import Any, Union
import torch

def to_tensor(x: Any, device: Union[str, torch.device] = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """将 array/tensor 转换为指定设备的 tensor"""
    return torch.as_tensor(x, device=device, dtype=dtype)
