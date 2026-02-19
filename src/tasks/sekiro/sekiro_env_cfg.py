from dataclasses import dataclass, field

import numpy as np
from src.gamelab.envs.manager_based_env_cfg import (
    SceneCfg, 
    RewardTermCfg, 
    ObservationTermCfg,
    TerminationTermCfg, 
    ActionTermCfg, 
    EventTermCfg,
    CommandTermCfg,
    RecorderTermCfg
)
from src.gamelab.managers.observation_manager import ObservationManager
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
        pos=config.cfg.scene.pos,
        capture_fps=config.cfg.scene.capture_fps,
        debug_vis_fps=config.cfg.ui.debug_vis_fps,
        assets={
            "sekiro": SekiroAssetCfg()
        }
    ))
    
    # 2. 观测项配置 (Observation Groups)
    observations: dict = field(default_factory=lambda: {
        "policy": ObservationTermCfg(
            func=sekiro_mdp.observations.image_frame,
            low=0,
            high=255,
            shape=(3, config.cfg.scene.img_height, config.cfg.scene.img_width),
            dtype=np.uint8
        ),
        "telemetry": ObservationTermCfg(
            func=sekiro_mdp.observations.memory_metrics,
            low=0,
            high=np.inf,
            scale=1/10000,
            shape=(10,),
            dtype=np.float32
        ),
    })
    
    # 3. 动作项配置 (Action Terms)
    actions: dict = field(default_factory=lambda: {
        "body": sekiro_mdp.actions.body_action_cfg([
            sekiro_mdp.actions.MOVE_MAP, 
            sekiro_mdp.actions.SKILL_MAP
        ])
    })

    # 3.5 事件项配置 (Event Terms)
    events: dict = field(default_factory=lambda: {
        "player_deaths": EventTermCfg(func=sekiro_mdp.events.player_death_event),
        "enemy_deaths": EventTermCfg(func=sekiro_mdp.events.enemy_death_event),
        "player_hp_decreased": EventTermCfg(func=sekiro_mdp.events.player_hp_decreased_event),
        "enemy_hp_decreased": EventTermCfg(func=sekiro_mdp.events.enemy_hp_decreased_event),
        "player_posture_changed": EventTermCfg(func=sekiro_mdp.events.player_posture_changed_event),
        "enemy_posture_changed": EventTermCfg(func=sekiro_mdp.events.enemy_posture_changed_event),
    })
    
    # 4. 奖励项配置 (Reward Terms)
    # 遵循 Isaac Lab 风格：直接使用 MDP 层提供的配置工厂，实现解耦且简洁
    rewards: dict = field(default_factory=lambda: {
        "player_death": RewardTermCfg(func=sekiro_mdp.rewards.player_death_reward,weight=1.0),
        "boss_death": RewardTermCfg(func=sekiro_mdp.rewards.boss_death_reward,weight=1.0),
        "player_health": RewardTermCfg(func=sekiro_mdp.rewards.player_health_reward,weight=0.2),
        "boss_health": RewardTermCfg(func=sekiro_mdp.rewards.boss_health_reward,weight=0.5),
        "player_posture": RewardTermCfg(func=sekiro_mdp.rewards.player_posture_reward,weight=0.2),
        "boss_posture": RewardTermCfg(func=sekiro_mdp.rewards.boss_posture_reward,weight=2.0),
        "survival": RewardTermCfg(func=sekiro_mdp.rewards.survival_reward,weight=1.0)
    })
    
    # 5. 终止项配置 (Termination Terms)
    terminations: dict = field(default_factory=lambda: {
        "player_dead": TerminationTermCfg(func=sekiro_mdp.terminations.player_dead_termination),
        "boss_dead": TerminationTermCfg(func=sekiro_mdp.terminations.boss_dead_termination),
        "time_out": TerminationTermCfg(func=sekiro_mdp.terminations.time_out_termination, time_out=True),
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

    # 7. 训练回调配置 (SB3 风格)
    # 允许在不修改 train.py 的情况下，通过配置挂载不同的观察/诊断工具
    callbacks: list = field(default_factory=lambda: [
        {
            "class": "src.tasks.sekiro.utils.SekiroCombinedCallback",   
            "params": {"log_interval": config.cfg.path.tb_log_interval}
        }
    ])
