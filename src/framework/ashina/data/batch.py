import numpy as np
import torch
from typing import Any, Dict, Optional, Union

class Batch:
    """
    天授式核心数据结构：Batch。
    它像字典一样存储数据，但支持类似 Tensor 的索引、切片和合并操作。
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, index: Union[int, slice, np.ndarray, torch.Tensor]) -> 'Batch':
        """支持对 Batch 内所有数据同步进行索引或切片"""
        new_batch = Batch()
        for k, v in self.__dict__.items():
            if hasattr(v, "__getitem__"):
                setattr(new_batch, k, v[index])
            else:
                setattr(new_batch, k, v)
        return new_batch

    def __len__(self) -> int:
        """返回 Batch 的大小（以第一个属性的长度为准）"""
        for v in self.__dict__.values():
            if hasattr(v, "__len__"):
                return len(v)
        return 0

    def to_torch(self, device: Optional[torch.device] = None) -> 'Batch':
        """将 Batch 内所有 numpy 数组转换为 torch Tensor"""
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                tensor = torch.from_numpy(v)
                if device:
                    tensor = tensor.to(device)
                setattr(self, k, tensor)
            elif isinstance(v, torch.Tensor) and device:
                setattr(self, k, v.to(device))
        return self

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __repr__(self) -> str:
        s = "Batch(\n"
        for k, v in self.__dict__.items():
            shape = getattr(v, "shape", "no shape")
            s += f"  {k}: {type(v).__name__} {shape},\n"
        s += ")"
        return s
