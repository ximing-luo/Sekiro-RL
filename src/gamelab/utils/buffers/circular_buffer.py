import torch
from typing import Optional

class CircularBuffer:
    """循环缓冲区，用于存储固定长度的历史张量数据。
    对标 Isaac Lab 的 CircularBuffer。
    """
    def __init__(self, capacity: int, num_envs: int, device: str, shape: tuple = ()):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.shape = shape
        
        # 初始化缓冲区 (capacity, num_envs, *shape)
        self.buffer = torch.zeros((capacity, num_envs, *shape), device=device)
        self.ptr = 0
        self.is_full = False

    def append(self, data: torch.Tensor):
        """添加新数据。
        Args:
            data: 形状为 (num_envs, *shape) 的张量。
        """
        self.buffer[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.is_full = True

    def get_all(self) -> torch.Tensor:
        """按时间顺序获取所有数据。
        Returns:
            形状为 (capacity, num_envs, *shape) 的张量。
        """
        if not self.is_full:
            return self.buffer[:self.ptr]
        
        # 重新排序使之按时间线性排列
        return torch.cat((self.buffer[self.ptr:], self.buffer[:self.ptr]), dim=0)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """重置指定环境的缓冲区。"""
        if env_ids is None:
            self.buffer.zero_()
            self.ptr = 0
            self.is_full = False
        else:
            self.buffer[:, env_ids] = 0.0
