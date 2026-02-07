"""
只狼特定的原子奖励函数。
这些函数尽量复用 mdp.common 中的通用逻辑，仅保留只狼特有的业务判断。
数值权重已移至 Config 层。
"""

import torch as th
from src.gamelab.envs import mdp

def player_death_reward(env, events, **kwargs):
    """自身死亡惩罚。"""
    # 0 为死亡事件，1 为击杀事件（防止同归于尽时的惩罚）
    return mdp.event_based_reward(env, event_id=0, reward_val=-10.0, events=events, **kwargs) if 1 not in events else 0.0

def boss_death_reward(env, events, **kwargs):
    """Boss 击杀奖励。"""
    return mdp.event_based_reward(env, event_id=1, reward_val=10.0, events=events, **kwargs)

def player_health_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """自身血量惩罚：只有掉血事件（2）发生时才计算。"""
    if 2 in events:
        # scale 0.01 表示 100 伤害 = 1.0 原始分
        return mdp.attribute_delta_reward(env, prev_metrics, next_metrics, metric_key='self_blood', scale=0.01, **kwargs)
    return 0.0

def boss_health_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """Boss 掉血奖励：只有掉血事件（5）发生时才计算。"""
    if 5 in events:
        # scale -0.01 因为 Boss 血量减少对 AI 是正奖励
        return mdp.attribute_delta_reward(env, prev_metrics, next_metrics, metric_key='boss_blood', scale=-0.01, **kwargs)
    return 0.0

def player_stamina_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """自身架势惩罚：只有架势恶化事件（6）发生时才计算。"""
    if 6 in events:
        # 只狼内存中架势值增加（条变长）代表恶化
        return mdp.attribute_delta_reward(env, prev_metrics, next_metrics, metric_key='self_stamina', scale=-0.01, **kwargs)
    return 0.0

def boss_stamina_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """Boss 架势奖励：只有架势增加事件（7）发生时才计算。"""
    if 7 in events:
        return mdp.attribute_delta_reward(env, prev_metrics, next_metrics, metric_key='boss_stamina', scale=0.01, **kwargs)
    return 0.0

def survival_reward(env, action, move_cost: float = -0.05, skill_cost: float = -0.05, **kwargs):
    """
    动作成本奖励：通过成本引导精准战斗。
    """
    if not isinstance(action, (list, th.Tensor)):
        return 0.0
        
    move_idx, skill_idx = action[0], action[1]
    reward = 0.0
    if move_idx != 0: reward += move_cost
    if skill_idx != 0: reward += skill_cost
    
    return reward / len(action)
