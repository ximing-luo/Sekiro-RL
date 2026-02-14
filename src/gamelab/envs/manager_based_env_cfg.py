from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from src.gamelab.managers.manager_term_cfg import (
    ObservationTermCfg,
    ActionTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    CommandTermCfg,
    RecorderTermCfg
)

from src.gamelab.assets.asset_base_cfg import AssetBaseCfg

@dataclass
class SceneCfg:
    """环境场景配置，对标 Isaac Lab 的 InteractiveSceneCfg。"""
    num_envs: int = 1
    observation_w: int = None
    observation_h: int = None
    pos: str = "offscreen"
    capture_fps: int = 60
    debug_vis_fps: int = 0
    # 资产列表
    assets: Dict[str, AssetBaseCfg] = field(default_factory=dict)

@dataclass
class ManagerBasedEnvCfg:
    """基础管理器驱动环境配置。"""
    scene: SceneCfg = field(default_factory=SceneCfg)
    # 管理器配置项（默认为空，由子类或任务配置填充）
    observations: Dict[str, ObservationTermCfg] = field(default_factory=dict)
    actions: Dict[str, ActionTermCfg] = field(default_factory=dict)
    events: Dict[str, EventTermCfg] = field(default_factory=dict)
    rewards: Dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations: Dict[str, TerminationTermCfg] = field(default_factory=dict)
    commands: Dict[str, CommandTermCfg] = field(default_factory=dict)
    recorders: Dict[str, RecorderTermCfg] = field(default_factory=dict)
    curriculums: Dict[str, Any] = field(default_factory=dict)
