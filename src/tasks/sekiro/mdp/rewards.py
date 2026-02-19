"""
只狼特定的原子奖励函数。
这些函数尽量复用 mdp.common 中的通用逻辑，仅保留只狼特有的业务判断。
数值权重已移至 Config 层。
"""
from __future__ import annotations
import torch as th
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv
    from src.gamelab.managers.manager_term_cfg import RewardTermCfg

def survival_reward(env: ManagerBasedEnv, cfg: RewardTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """只狼生存/动作成本奖励。"""
    rewards = th.zeros(env.num_envs, device=env.device)

    # 从 ActionManager 获取最新动作
    action = env.action_manager.action
    if action is None: return rewards

    # 针对 MultiDiscrete 动作 [num_envs, action_dim]
    move_indices = action[:, 0]
    skill_indices = action[:, 1]

    move_cost: float = -0.05
    skill_cost: float = -0.2
    rewards[move_indices != 0] += move_cost
    rewards[skill_indices != 0] += skill_cost
    
    return rewards

def player_death_reward(env: ManagerBasedEnv, cfg: RewardTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """玩家死亡奖励。"""
    rewards = th.zeros(env.num_envs, device=env.device)
    data = env.scene.sekiro.status
    player_deads = data.player_deaths
    prev_player_deads = data.prev_player_deaths
    # 只要当前死亡次数大于上一帧，即视为发生死亡事件
    death_mask = player_deads > prev_player_deads
    rewards[death_mask] = -10.0
    if rewards.any(): print(f"\033[34mrewards: 玩家死亡触发: {rewards[death_mask].tolist()}\033[0m")
    return rewards 

def boss_death_reward(env: ManagerBasedEnv, cfg: RewardTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """Boss死亡奖励。"""
    rewards = th.zeros(env.num_envs, device=env.device)
    data = env.scene.sekiro.status
    enemy_deads = data.enemy_deaths
    prev_enemy_deads = data.prev_enemy_deaths
    # 只要当前Boss死亡次数大于上一帧，即视为发生击杀事件
    death_mask = enemy_deads > prev_enemy_deads
    rewards[death_mask] = 10.0
    if rewards.any(): print(f"\033[31mrewards: BOSS死亡触发: {rewards[death_mask].tolist()}\033[0m")
    return rewards

def player_health_reward(env: ManagerBasedEnv, cfg: RewardTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """玩家血量奖励。"""
    rewards = th.zeros(env.num_envs, device=env.device)
    data = env.scene.sekiro.player
    hp = data.hp
    prev_hp = data.prev_hp
    # 血量变化 = 当前血量 - 上一帧血量
    hp_delta = hp - prev_hp
    if hp_delta < 0: rewards += hp_delta * 0.01  # 假设每掉 100 点血量奖励 1
    if rewards.any(): print(f"\033[34mrewards: 玩家血量变化: {rewards[rewards != 0].tolist()}\033[0m")
    return rewards

def boss_health_reward(env: ManagerBasedEnv, cfg: RewardTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """Boss血量奖励。"""
    rewards = th.zeros(env.num_envs, device=env.device)
    data = env.scene.sekiro.enemy
    hp = data.hp
    prev_hp = data.prev_hp
    # 血量变化 = 当前血量 - 上一帧血量
    hp_delta = hp - prev_hp
    if hp_delta < 0: rewards += -hp_delta * 0.01  # 假设每掉 100 点血量奖励 1
    if rewards.any(): print(f"\033[31mrewards: BOSS血量变化: {rewards[rewards != 0].tolist()}\033[0m")
    return rewards

def player_posture_reward(env: ManagerBasedEnv, cfg: RewardTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor:
    """玩家架势奖励。"""
    rewards = th.zeros(env.num_envs, device=env.device)
    data = env.scene.sekiro.player
    posture = data.posture
    prev_posture = data.prev_posture
    # 架势变化 = 当前架势 - 上一帧架势
    posture_delta = posture - prev_posture
    if posture_delta < 0: rewards += posture_delta * 0.01  # 假设每改变 100 点架势奖励 1
    if rewards.any(): print(f"\033[34mrewards: 玩家架势条变化: {rewards[rewards != 0].tolist()}\033[0m")
    return rewards 

def boss_posture_reward(env: ManagerBasedEnv, cfg: RewardTermCfg, env_ids: Sequence[int] | None = None) -> th.Tensor: 
    """Boss架势奖励。"""
    rewards = th.zeros(env.num_envs, device=env.device)
    data = env.scene.sekiro.enemy
    posture = data.posture
    prev_posture = data.prev_posture
    # 架势变化 = 当前架势 - 上一帧架势
    posture_delta = posture - prev_posture
    if posture_delta < 0: rewards += -posture_delta * 0.01  # 假设每改变 100 点架势奖励 1
    if rewards.any(): print(f"\033[31mrewards: BOSS架势条变化: {rewards[rewards != 0].tolist()}\033[0m")
    return rewards