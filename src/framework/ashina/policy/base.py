from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from ..data.batch import Batch

class BasePolicy(nn.Module, ABC):
    """
    策略基类。天授设计思路：
    1. forward: 处理 Batch 输入，返回包含 logits/act 的 Batch。
    2. learn: 处理从 Buffer 采样的 Batch，进行学习更新。
    """
    def __init__(self, action_dim, device="cuda"):
        super().__init__()
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, batch: Batch, state: Optional[Any] = None) -> Batch:
        """输出动作信息"""
        pass

    @abstractmethod
    def learn(self, batch: Batch) -> Dict[str, float]:
        """学习更新，返回 loss 字典"""
        pass
