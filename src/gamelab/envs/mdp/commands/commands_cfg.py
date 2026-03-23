from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class CommandTermCfg:
    """指令项配置。"""
    func: Callable
    params: dict = None
    resampling_time: float = 0.0 # 重新采样的时间间隔
