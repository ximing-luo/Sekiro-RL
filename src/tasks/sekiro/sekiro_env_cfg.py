from dataclasses import dataclass, field
from src.envs.manager_based_env_cfg import SceneCfg, RewardTermCfg, ObservationTermCfg, TerminationTermCfg, ActionTermCfg
from src.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
import src.envs.mdp as mdp
import configs.config as config

@dataclass
class SekiroEnvCfg(ManagerBasedRLEnvCfg):
    """
    只狼环境的总配置类。
    通过定义各个管理器的 Terms 来实现真正的逻辑抽象。
    """
    
    # 1. 场景配置
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(
        observation_w=config.cfg.scene.img_width,
        observation_h=config.cfg.scene.img_height,
        pos="top_left",
        capture_fps=config.cfg.scene.capture_fps,
        debug_vis_fps=config.cfg.ui.debug_vis_fps
    ))
    
    # 2. 观测项配置 (Observation Terms)
    observations: dict = field(default_factory=lambda: {
        "policy": ObservationTermCfg(func=mdp.observations.image_frame), # 图像观测
        "telemetry": ObservationTermCfg(func=mdp.observations.memory_metrics), # 数值指标
    })
    
    # 3. 动作项配置 (Action Terms)
    actions: dict = field(default_factory=lambda: {
        "body": ActionTermCfg(func=mdp.actions.sekiro_multi_discrete_action, params={"dims": mdp.actions.MULTI_DISCRETE_DIMS})
    })
    
    # 4. 奖励项配置 (Reward Terms)
    # rewards: dict = field(default_factory=lambda: {
    #     "player_death": RewardTermCfg(func=mdp.rewards.player_death_reward, weight=0.5), # 最终分: -5.0
    #     "boss_death": RewardTermCfg(func=mdp.rewards.boss_death_reward, weight=1.0), # 最终分: +10.0
    #     "player_health": RewardTermCfg(func=mdp.rewards.player_health_reward, weight=0.2), # 100伤害 = -0.2
    #     "boss_health": RewardTermCfg(func=mdp.rewards.boss_health_reward, weight=0.4), # 100伤害 = +0.4
    #     "player_stamina": RewardTermCfg(func=mdp.rewards.player_stamina_reward, weight=0.2), # 100上涨 = -0.2
    #     "boss_stamina": RewardTermCfg(func=mdp.rewards.boss_stamina_reward, weight=0.5), # 100下降 = +0.5
    #     "survival": RewardTermCfg(func=mdp.rewards.survival_reward, weight=0.2), # 动作/时间正则化
    # })

    rewards: dict = field(default_factory=lambda: {
        "player_health": RewardTermCfg(func=mdp.rewards.player_health_reward, weight=0.2), # 100伤害 = -0.2
        "boss_stamina": RewardTermCfg(func=mdp.rewards.boss_stamina_reward, weight=0.3), # 100下降 = +0.5
        "survival": RewardTermCfg(func=mdp.rewards.survival_reward, weight=0.2), # 动作/时间正则化
    })
    
    # 5. 终止项配置 (Termination Terms)
    terminations: dict = field(default_factory=lambda: {
        "player_dead": TerminationTermCfg(func=mdp.terminations.player_dead_termination),
        "boss_dead": TerminationTermCfg(func=mdp.terminations.boss_dead_termination),
    })

    # 6. RL 基础配置
    buffer_size: int = config.cfg.rl.buffer_size
    frame_history_len: int = config.cfg.rl.frame_history_len
    n_step_rewards: int = config.cfg.rl.n_step_rewards
