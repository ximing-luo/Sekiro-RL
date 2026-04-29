import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

class BaseNoise(ABC, object):
    """
    动作噪声基类。
    """
    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """重置到初始状态。"""
        pass

    @abstractmethod
    def __call__(self, size: Sequence[int]) -> np.ndarray:
        """生成新的噪声。"""
        raise NotImplementedError

class GaussianNoise(BaseNoise):
    """
    高斯噪声，常用于 DDPG、TD3 等连续动作空间的探索。
    """
    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        super().__init__()
        self._mu = mu
        assert 0 <= sigma, "Noise std should not be negative."
        self._sigma = sigma

    def __call__(self, size: Sequence[int]) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma, size)

class OUNoise(BaseNoise):
    """
    Ornstein-Uhlenbeck (OU) 噪声。
    一种带有均值回归特性的时间相关噪声过程，常用于 DDPG 连续动作空间的探索。
    在物理系统或存在惯性的环境（如驾驶、机械臂）中，OU 噪声能提供时间上更平滑的探索轨迹。
    """
    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.3,
        theta: float = 0.15,
        dt: float = 1e-2,
        x0: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self._mu = mu
        self._alpha = theta * dt
        self._beta = sigma * np.sqrt(dt)
        self._x0 = x0
        self._x: Optional[Union[float, np.ndarray]] = None
        self.reset()

    def reset(self, size: Optional[Sequence[int]] = None) -> None:
        """重置状态。通常在每个 episode 开始时调用。"""
        if size is None:
            self._x = self._x0
        else:
            self._x = np.zeros(size) if self._x0 is None else np.full(size, float(self._x0))

    def __call__(self, size: Sequence[int], mu: Optional[float] = None) -> np.ndarray:
        if self._x is None:
            self._x = np.zeros(size)
        if mu is None:
            mu = self._mu
        r = self._beta * np.random.normal(size=size)
        self._x = self._x + self._alpha * (mu - self._x) + r
        return self._x  # type: ignore
