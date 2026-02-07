from dataclasses import dataclass, field
from src.gamelab.envs.manager_based_env_cfg import (
    SceneCfg, 
    RewardTermCfg, 
    ObservationTermCfg, 
    ObservationGroupCfg,
    TerminationTermCfg, 
    ActionTermCfg, 
    EventTermCfg,
    CommandTermCfg,
    RecorderTermCfg
)
from src.gamelab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from src.gamelab.assets.sekiro.sekiro_asset_cfg import SekiroAssetCfg
import src.gamelab.envs.mdp as mdp
import src.tasks.sekiro.mdp as sekiro_mdp
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
        debug_vis_fps=config.cfg.ui.debug_vis_fps,
        assets={
            "player": SekiroAssetCfg()
        }
    ))
    
    # 2. 观测项配置 (Observation Groups)
    observations: dict = field(default_factory=lambda: {
        "policy": ObservationGroupCfg(
            concatenate_terms=True,
            terms={
                "image": ObservationTermCfg(func=sekiro_mdp.observations.image_frame),
                "telemetry": ObservationTermCfg(func=sekiro_mdp.observations.memory_metrics),
            }
        )
    })
    
    # 3. 动作项配置 (Action Terms)
    actions: dict = field(default_factory=lambda: {
        "body": ActionTermCfg(func=sekiro_mdp.actions.sekiro_multi_discrete_action, params={"dims": sekiro_mdp.actions.MULTI_DISCRETE_DIMS})
    })

    # 3.5 事件项配置 (Event Terms)
    events: dict = field(default_factory=lambda: {
        "sekiro_events": EventTermCfg(func=sekiro_mdp.events.sekiro_events_logic)
    })
    
    # 4. 奖励项配置 (Reward Terms)
    # 遵循 Isaac Lab 风格：在这里调整权重(weight)和参数(params)，不要改 mdp 源码
    rewards: dict = field(default_factory=lambda: {
        "player_death": RewardTermCfg(func=sekiro_mdp.rewards.player_death_reward, weight=1.0), # 原始分 -10.0 -> 最终 -10.0
        "boss_death": RewardTermCfg(func=sekiro_mdp.rewards.boss_death_reward, weight=1.0),     # 原始分 +10.0 -> 最终 +10.0
        "player_health": RewardTermCfg(func=sekiro_mdp.rewards.player_health_reward, weight=0.2), # 100伤害 = -1.0 * 0.2 = -0.2
        "boss_health": RewardTermCfg(func=sekiro_mdp.rewards.boss_health_reward, weight=0.5),   # 100伤害 = +1.0 * 0.5 = +0.5
        "player_stamina": RewardTermCfg(func=sekiro_mdp.rewards.player_stamina_reward, weight=0.2), # 100恶化 = -1.0 * 0.2 = -0.2
        "boss_stamina": RewardTermCfg(func=sekiro_mdp.rewards.boss_stamina_reward, weight=0.5), # 100进度 = +1.0 * 0.5 = +0.5
        "survival": RewardTermCfg(
            func=sekiro_mdp.rewards.survival_reward, 
            weight=1.0, 
            params={"move_cost": -0.05, "skill_cost": -0.02} # 在这里调动作成本
        ),
    })
    
    # 5. 终止项配置 (Termination Terms)
    terminations: dict = field(default_factory=lambda: {
        "player_dead": TerminationTermCfg(func=sekiro_mdp.terminations.player_dead_termination),
        "boss_dead": TerminationTermCfg(func=sekiro_mdp.terminations.boss_dead_termination),
    })

    # 5.5 指令项配置 (Command Terms)
    # 对标 Isaac Lab，用于提供高层目标（如：积极进攻、保持距离）
    commands: dict = field(default_factory=lambda: {
        "base_command": CommandTermCfg(func=mdp.commands.null_command)
    })

    # 5.6 记录项配置 (Recorder Terms)
    recorders: dict = field(default_factory=lambda: {
        "basic": RecorderTermCfg(func=mdp.recorders.basic_recorder)
    })

    # 6. RL 基础配置
    buffer_size: int = config.cfg.rl.buffer_size
    frame_history_len: int = config.cfg.rl.frame_history_len
    n_step_rewards: int = config.cfg.rl.n_step_rewards

    # 7. 训练回调配置 (SB3 风格)
    # 允许在不修改 train.py 的情况下，通过配置挂载不同的观察/诊断工具
    callbacks: list = field(default_factory=lambda: [
        {
            "class": "src.gamelab.utils.rl.sb3.SekiroCombinedCallback",
            "params": {"log_interval": 1000}
        }
    ])
