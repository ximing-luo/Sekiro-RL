from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch

@dataclass
class ManagerTermBaseCfg:
    """管理器术语的基础配置。"""
    func: Callable[..., Any] = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RewardTermCfg(ManagerTermBaseCfg):
    """奖励项配置。"""
    weight: float = 1.0

@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """观测项配置。"""
    scale: float = 1.0
    noise: Optional[Any] = None
    clip: Optional[Tuple[float, float]] = None

@dataclass
class ObservationGroupCfg:
    """观测组配置。"""
    concatenate_terms: bool = True
    terms: Dict[str, ObservationTermCfg] = field(default_factory=dict)

@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
    """终止项配置。"""
    time_out: bool = False

@dataclass
class EventTermCfg(ManagerTermBaseCfg):
    """事件项配置。"""
    mode: str = "reset"  # 'reset', 'interval', 'startup'

@dataclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    """课程项配置。"""
    pass

@dataclass
class CommandTermCfg:
    """指令项配置。"""
    func: Callable[..., Any] = None
    params: Dict[str, Any] = field(default_factory=dict)
