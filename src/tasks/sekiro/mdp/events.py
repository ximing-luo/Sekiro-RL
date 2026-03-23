"""
只狼特定的事件检测。
"""
from __future__ import annotations
import torch as th
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv
    from src.gamelab.managers.manager_term_cfg import EventTermCfg

def player_death_event(env: ManagerBasedEnv, cfg: EventTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """玩家死亡事件"""
    event = th.full((env.num_envs,), -1, device=env.device)
    data = env.scene.sekiro.status
    player_deads = data.player_deaths
    prev_player_deads = data.prev_player_deaths
    event = th.where(player_deads > prev_player_deads, 0, event)
    return event

def enemy_death_event(env: ManagerBasedEnv, cfg: EventTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """Boss死亡事件"""
    event = th.full((env.num_envs,), -1, device=env.device)
    data = env.scene.sekiro.status
    enemy_deads = data.enemy_deaths
    prev_enemy_deads = data.prev_enemy_deaths
    event = th.where(enemy_deads > prev_enemy_deads, 1, event)
    return event

def player_hp_decreased_event(env: ManagerBasedEnv, cfg: EventTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """玩家掉血事件"""
    event = th.full((env.num_envs,), -1, device=env.device)
    data = env.scene.sekiro.status
    player_hps = data.player_hp
    prev_player_hps = data.prev_player_hp
    event = th.where(player_hps < prev_player_hps, 2, event)
    return event

def enemy_hp_decreased_event(env: ManagerBasedEnv, cfg: EventTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """Boss掉血事件"""
    event = th.full((env.num_envs,), -1, device=env.device)
    data = env.scene.sekiro.status
    enemy_hps = data.enemy_hp
    prev_enemy_hps = data.prev_enemy_hp
    event = th.where(enemy_hps < prev_enemy_hps, 3, event)
    return event

def player_posture_changed_event(env: ManagerBasedEnv, cfg: EventTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """玩家架势变化事件"""
    event = th.full((env.num_envs,), -1, device=env.device)
    data = env.scene.sekiro.status
    posture = data.posture
    prev_posture = data.prev_posture
    event = th.where(posture != prev_posture, 4, event)
    return event

def enemy_posture_changed_event(env: ManagerBasedEnv, cfg: EventTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """Boss架势变化事件"""
    event = th.full((env.num_envs,), -1, device=env.device)
    data = env.scene.sekiro.status
    posture = data.posture
    prev_posture = data.prev_posture
    event = th.where(posture != prev_posture, 5, event)
    return event