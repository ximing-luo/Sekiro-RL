from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

@dataclass
class ManagerTermCfg:
    """基础项配置类，对应 Isaac Lab 中的 ManagerTermCfg。"""
    func: Callable  # 处理逻辑的函数
    params: Dict[str, Any] = field(default_factory=dict)  # 函数参数

@dataclass
class RewardTermCfg(ManagerTermCfg):
    """奖励项配置。"""
    weight: float = 1.0  # 奖励权重

@dataclass
class ObservationTermCfg(ManagerTermCfg):
    """观测项配置。"""
    # 可以在此扩展如 noise, clip 等配置
    pass

@dataclass
class TerminationTermCfg(ManagerTermCfg):
    """终止项配置。"""
    is_terminal: bool = True  # 是否触发回合结束

@dataclass
class ActionTermCfg(ManagerTermCfg):
    """动作项配置。"""
    pass

@dataclass
class SceneCfg:
    observation_w: int = 640
    observation_h: int = 360
    pos: str = "offscreen"
    capture_fps: int = 60
    debug_vis_fps: int = 0

@dataclass
class ManagerBasedEnvCfg:
    """基础管理器驱动环境配置。"""
    scene: SceneCfg = field(default_factory=SceneCfg)
    # 管理器配置项（默认为空，由子类或任务配置填充）
    observations: Dict[str, ObservationTermCfg] = field(default_factory=dict)
    actions: Dict[str, ActionTermCfg] = field(default_factory=dict)
    rewards: Dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations: Dict[str, TerminationTermCfg] = field(default_factory=dict)
