from abc import ABC, abstractmethod

class BaseBuffer(ABC):
    """
    基础 Buffer 类，定义了所有 Buffer 应该实现的接口。
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        
    @abstractmethod
    def add(self, *args, **kwargs):
        """添加一条经验。"""
        pass
        
    @abstractmethod
    def sample(self, batch_size: int):
        """采样一个批次的经验。"""
        pass
        
    @abstractmethod
    def __len__(self):
        """返回当前 Buffer 中的经验数量。"""
        pass
