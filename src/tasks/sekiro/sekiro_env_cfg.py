from dataclasses import dataclass, field
from src.envs.manager_based_env_cfg import SceneCfg
from src.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

@dataclass
class SekiroRewardCfg:
    """奖励权重配置。"""
    self_blood_gamma: float = 0.6
    boss_blood_gamma: float = 0.4
    self_stamina_gamma: float = 0.5
    boss_stamina_gamma: float = 0.5
    death_penalty: float = -200.0
    victory_reward: float = 200.0

@dataclass
class SekiroEnvCfg(ManagerBasedRLEnvCfg):
    """只狼环境的总配置类。"""
    
    # 覆盖默认场景配置
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(
        observation_w=640,
        observation_h=360,
        pos="offscreen",
        capture_fps=60,
        debug_vis_fps=0
    ))
    
    # 任务特有配置
    rewards: SekiroRewardCfg = field(default_factory=SekiroRewardCfg)
    
    # 继承的 RL 配置
    buffer_size: int = 50000
    frame_history_len: int = 4
    n_step_rewards: int = 20
