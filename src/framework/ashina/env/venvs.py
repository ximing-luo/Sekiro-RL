from abc import ABC, abstractmethod
from typing import Any

class VectorEnv(ABC):
    """
    向量化环境基类。
    框架只定义接口，不关心具体实现（GPU/CPU/IsaacLab/自定义）。
    """
    @abstractmethod
    def reset(self, **kwargs: Any) -> Any:
        """返回 (obs_batch[N, ...], info_list)"""
        pass

    @abstractmethod
    def step(self, actions: Any) -> Any:
        """
        返回 (obs_batch, rew_batch, done_batch, trunc_batch, info_list)
        内部自动处理 done 后 reset，info["terminal_obs"] 保留终止帧
        """
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Any:
        pass
