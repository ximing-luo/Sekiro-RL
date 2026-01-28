from abc import ABC, abstractmethod
import torch
import os

class BaseAgent(ABC):
    """
    RL 代理基类，定义了算法的标准接口。
    借鉴了 Stable-Baselines3 和 SKRL 的设计理念。
    """
    def __init__(self, action_dim, device="cuda"):
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # 移除可能引起冲突的属性初始化，由具体子类或算法类管理

    @abstractmethod
    def act(self, state, epsilon=0.0):
        """
        根据当前状态选择动作。
        """
        pass

    @abstractmethod
    def record(self, state, action, reward, next_state, done):
        """
        将经验存入缓冲区（可选，有些算法可能由 Runner 负责）。
        """
        pass

    @abstractmethod
    def learn(self):
        """
        执行一轮训练优化。
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        保存模型。
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        加载模型。
        """
        pass

    def to_device(self, *tensors):
        """便捷方法：将张量移至指定设备。"""
        return [t.to(self.device) for t in tensors]
