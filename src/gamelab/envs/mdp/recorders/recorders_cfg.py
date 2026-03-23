from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class RecorderTermCfg:
    """记录项配置。"""
    func: Callable
    params: Dict[str, Any] = None
