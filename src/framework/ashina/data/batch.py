import numpy as np
import torch
from typing import Any, Dict, Optional, Union

class Batch:
    """
    核心数据结构：Batch。
    它像字典一样存储数据，但支持类似 Tensor 的索引、切片和合并操作。
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, index: Union[int, slice, np.ndarray, torch.Tensor]) -> 'Batch':
        """同步索引：所有属性必须支持索引（符合 RL Batch 定义）"""
        return Batch(**{k: v[index] for k, v in self.__dict__.items()})

    def __len__(self) -> int:
        """返回 Batch 大小：显式迭代"""
        for v in self.__dict__.values():
            return len(v)
        return 0

    def to_torch(self, device: Optional[torch.device] = None) -> 'Batch':
        """确定性转换：利用 torch.as_tensor 自动分发"""
        for k, v in self.__dict__.items():
            # 自动处理 numpy 和 tensor
            t = torch.as_tensor(v, device=device)
            # 强行收敛浮点精度
            if t.dtype == torch.float64:
                t = t.to(torch.float32)
            self.__dict__[k] = t
        return self

    def __getattr__(self, key: str) -> Any:
        """
        属性访问保护。
        如果属性不存在，返回 None 而不是抛出异常，
        这在处理某些算法可选的 Batch 键（如 weights, info）时非常优雅。
        """
        return self.__dict__.get(key, None)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __repr__(self) -> str:
        s = "Batch(\n"
        for k, v in self.__dict__.items():
            if k.startswith("_"): continue
            shape = getattr(v, "shape", "no shape")
            s += f"  {k}: {type(v).__name__} {shape},\n"
        s += ")"
        return s
