from abc import ABC, abstractmethod
import numpy as np

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
