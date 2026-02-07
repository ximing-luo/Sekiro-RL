

import torch
import numpy as np
from typing import Any, Union

def to_torch(data: Any, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """将各种数据转换为 Torch 张量。"""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device=device, dtype=dtype)
    return torch.tensor(data, device=device, dtype=dtype)

def to_numpy(data: Any) -> np.ndarray:
    """将数据转换为 Numpy 数组。"""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)
