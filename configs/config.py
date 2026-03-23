from dataclasses import dataclass, field

@dataclass(frozen=True)
class SceneConfig:
    """基础环境与采集配置。"""
    img_width: int = 240
    img_height: int = 128
    capture_fps: int = 60
    target_fps: int = 60
    camera_index: int = 1
    camera_width: int = 1920
    camera_height: int = 1080
    pos: str = "top_right"
    num_envs: int = 1

@dataclass(frozen=True)
class UIConfig:
    """内存映射与 UI 窗口配置。"""
    debug_vis_fps: int = 30

@dataclass(frozen=True)
class RLConfig:
    """强化学习基础参数。"""

@dataclass(frozen=True)
class TrainConfig:
    """训练超参数 (PPO/SB3)。"""
    # -- 基础
    learning_rate: float = 2e-5
    gamma: float = 0.91
    batch_size: int = 64
    save_freq: int = 8192+2048
    steps: int = 2048 * 50
    # -- PPO 特有
    n_steps: int = 2048
    n_epochs: int = 5
    target_kl: float = 0.2
    ent_coef: float = 0.01
    clip_range: float = 0.3
    clip_range_vf: float = 0.3
    max_grad_norm: float = 5.0
    vf_coef: float = 0.5
    # -- 辅助任务
    aux_coef: float = 0.01

@dataclass(frozen=True)
class EpsilonConfig:
    """探索策略参数 (Epsilon Greedy)。"""
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 10000

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
    tb_log_interval: int = 4096

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
