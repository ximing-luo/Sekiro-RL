from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class SceneConfig:
    """基础环境与采集配置。"""
    img_width: int = 240 # 从 480 降至 240
    img_height: int = 135 # 从 270 降至 135
    capture_fps: int = 60
    target_fps: int = 60
    camera_index: int = 1
    camera_width: int = 1920
    camera_height: int = 1080
    pos: str = "offscreen"

@dataclass(frozen=True)
class UIConfig:
    """内存映射与 UI 窗口配置。"""
    blood_window: Tuple[int, int, int, int] = (110, 90, 625, 907)
    stamina_window: Tuple[int, int, int, int] = (586, 54, 750, 900)
    debug_vis_fps: int = 60

@dataclass(frozen=True)
class RLConfig:
    """强化学习基础参数。"""
    frame_history_len: int = 4
    buffer_size: int = 0 # 设置为 0 以在 PPO 训练中关闭冗余的 ReplayBuffer
    n_step_rewards: int = 5

@dataclass(frozen=True)
class TrainConfig:
    """训练超参数。"""
    lr: float = 0.0001
    gamma: float = 0.99
    batch_size: int = 8
    micro_batch_size: int = 2
    grad_clip_norm: float = 10.0
    optimize_every_steps: int = 5
    target_update_freq: int = 25
    save_freq: int = 100

@dataclass(frozen=True)
class EpsilonConfig:
    """探索策略参数 (Epsilon Greedy)。"""
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay: int = 50000

@dataclass(frozen=True)
class PERConfig:
    """优先经验回放 (PER) 参数。"""
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_steps: int = 200000

@dataclass(frozen=True)
class PathConfig:
    """路径与日志配置。"""
    model_path: str = "outputs/models/dqn_model.pth"
    log_dir: str = "logs"
    tb_log_interval: int = 2000

@dataclass(frozen=True)
class GlobalConfig:
    """全局总配置类。"""
    scene: SceneConfig = field(default_factory=SceneConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    epsilon: EpsilonConfig = field(default_factory=EpsilonConfig)
    per: PERConfig = field(default_factory=PERConfig)
    path: PathConfig = field(default_factory=PathConfig)

# 实例化全局配置对象，供其他模块使用
cfg = GlobalConfig()
