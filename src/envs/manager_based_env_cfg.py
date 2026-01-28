from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SceneCfg:
    observation_w: int = 640
    observation_h: int = 360
    pos: str = "offscreen"
    capture_fps: int = 60
    debug_vis_fps: int = 0

@dataclass
class ManagerBasedEnvCfg:
    scene: SceneCfg = field(default_factory=SceneCfg)
    # 子类可以扩展其他管理器的配置
