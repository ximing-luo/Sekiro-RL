from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import numpy as np

@dataclass
class ManagerTermBaseCfg:
    """管理器术语的基础配置。"""
    func: Callable[..., Any] = None
    class_type: type = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RewardTermCfg(ManagerTermBaseCfg):
    """奖励项配置。"""
    weight: float = 1.0

@dataclass
class DeltaRewardTermCfg(RewardTermCfg):
    """增量奖励项配置。"""
    group_name: str = "policy"
    term_name: str = "telemetry"
    dim: Optional[int] = None
    scale: float = 1.0

@dataclass
class EventRewardTermCfg(RewardTermCfg):
    """事件奖励项配置。"""
    event_id: int = 0
    reward: float = 1.0

@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """观测项配置。"""
    shape: Tuple[int, ...] = field(default_factory=tuple) # 必须提供形状
    low: Union[float, np.ndarray] = -np.inf
    high: Union[float, np.ndarray] = np.inf
    dtype: Any = np.float32
    scale: float = 1.0
    

@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
    """终止项配置。"""
    time_out: bool = False
    

@dataclass
class EventTermCfg(ManagerTermBaseCfg):
    """事件项配置。"""
    mode: str = "reset"  # 'reset', 'interval', 'startup'
    event_id: Optional[Any] = None
    

@dataclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    """课程项配置。"""
    

@dataclass
class ActionTermCfg(ManagerTermBaseCfg):
    """动作项配置。"""
    

@dataclass
class MultiDiscreteActionTermCfg(ActionTermCfg):
    """多维离散动作项配置。"""
    action_maps: List[Dict[int, Callable]] = field(default_factory=list)

@dataclass
class RecorderTermCfg(ManagerTermBaseCfg):
    """记录项配置。"""
    

@dataclass
class CommandTermCfg(ManagerTermBaseCfg):
    """指令项配置。"""
    
