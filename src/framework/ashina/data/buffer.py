from abc import ABC, abstractmethod
import numpy as np
from .batch import Batch


class BaseBuffer(ABC):
    def __init__(self, capacity: int):
        self.capacity = capacity

    @abstractmethod
    def add(self, obs: np.ndarray, act, rew: float, obs_next: np.ndarray, done: bool) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class ReplayBuffer(BaseBuffer):
    """
    经验回放缓冲区（环形数组实现）。
    存储 (obs, act, rew, obs_next, done) 五元组，延迟分配内存。
    """
    def __init__(self, size: int):
        super().__init__(size)
        self._obs = None
        self._obs_next = None
        self._act = None
        self._rew = None
        self._done = None
        self._ptr = 0
        self._len = 0

    def add(self, obs: np.ndarray, act, rew: float, obs_next: np.ndarray, done: bool) -> None:
        if self._obs is None:
            obs = np.asarray(obs)
            self._obs = np.zeros((self.capacity, *obs.shape), dtype=obs.dtype)
            self._obs_next = np.zeros_like(self._obs)
            self._act = np.zeros(self.capacity, dtype=np.int64)
            self._rew = np.zeros(self.capacity, dtype=np.float32)
            self._done = np.zeros(self.capacity, dtype=np.bool_)

        self._obs[self._ptr] = obs
        self._obs_next[self._ptr] = obs_next
        self._act[self._ptr] = act
        self._rew[self._ptr] = rew
        self._done[self._ptr] = done

        self._ptr = (self._ptr + 1) % self.capacity
        self._len = min(self._len + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        indices = np.random.choice(self._len, min(batch_size, self._len), replace=True)
        return Batch(
            obs=self._obs[indices],
            act=self._act[indices],
            rew=self._rew[indices],
            obs_next=self._obs_next[indices],
            done=self._done[indices],
        )

    def __len__(self) -> int:
        return self._len
