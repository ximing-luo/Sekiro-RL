from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BasePolicy(nn.Module, ABC):
    """
    策略基类，定义了策略的标准接口。
    只负责动作生成和 Loss 计算。
    """
    def __init__(self, action_dim, device="cuda"):
        super().__init__()
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, obs):
        """输出动作分布或 Q 值"""
        pass

    @abstractmethod
    def update(self, batch_data):
        """根据采样数据更新策略，返回 loss 信息"""
        pass

    def to_device(self, *tensors):
        return [t.to(self.device) for t in tensors]
