from abc import ABC, abstractmethod
from typing import Any

class BaseSensor(ABC):
    """
    传感器基类，定义从仿真器（游戏）获取数据的标准接口。
    """
    @abstractmethod
    def start(self):
        """启动传感器采集。"""
        pass

    @abstractmethod
    def stop(self):
        """停止传感器采集。"""
        pass

    @abstractmethod
    def get_data(self) -> Any:
        """获取传感器当前数据。"""
        pass
