import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union

class ExplorationStrategy(ABC):
    """
    探索策略基类。
    """
    @abstractmethod
    def add_noise(self, action: Any, **kwargs: Any) -> Any:
        pass

class EpsilonGreedy(ExplorationStrategy):
    """
    Epsilon-Greedy 探索策略，常用于离散动作空间。
    """
    def __init__(self, epsilon: float, action_dim: int):
        self.epsilon = epsilon
        self.action_dim = action_dim

    def add_noise(self, action: Any, **kwargs: Any) -> Any:
        if np.random.rand() < self.epsilon:
            # 使用 np.shape 替代类型检查
            # 这种方式更接近“物理化”的维度逻辑
            shape = np.shape(action)
            if shape:
                return np.random.randint(0, self.action_dim, size=shape)
            return np.random.randint(0, self.action_dim)
        return action

class GaussianNoise(ExplorationStrategy):
    """
    高斯噪声探索策略，常用于连续动作空间。
    """
    def __init__(self, sigma: float):
        self.sigma = sigma

    def add_noise(self, action: np.ndarray, **kwargs: Any) -> np.ndarray:
        return action + np.random.normal(0, self.sigma, size=action.shape)
