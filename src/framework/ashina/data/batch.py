import numpy as np
import torch
from typing import Any, Optional, Union

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
        first = next(iter(self.__dict__.values()), None)
        return len(first) if first is not None else 0

    def to_torch(self, device: Optional[torch.device] = None) -> 'Batch':
        for k, v in self.__dict__.items():
            t = torch.as_tensor(v, device=device)
            if t.dtype == torch.float64:
                t = t.to(torch.float32)
            self.__dict__[k] = t
        return self

    def __getattr__(self, key: str) -> Any:
        return object.__getattribute__(self, '__dict__').get(key, None)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    @staticmethod
    def cat(batches: list) -> 'Batch':
        """合并多个 Batch 为一个大 Batch（沿第 0 维拼接）"""
        if not batches:
            return Batch()
        merged = {}
        for key in batches[0].__dict__:
            values = [b.__dict__[key] for b in batches]
            if isinstance(values[0], torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], np.ndarray):
                merged[key] = np.concatenate(values, axis=0)
            else:
                merged[key] = values[0]
        return Batch(**merged)

    def __repr__(self) -> str:
        s = "Batch(\n"
        for k, v in self.__dict__.items():
            if k.startswith("_"): continue
            shape = getattr(v, "shape", "no shape")
            s += f"  {k}: {type(v).__name__} {shape},\n"
        s += ")"
        return s
