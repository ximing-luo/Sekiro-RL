"""
模块用途：原子奖励函数定义（Reward Terms）。
遵循 Isaac Lab 风格，每个函数只负责一个维度的奖励计算。
"""

def player_death_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """自身死亡惩罚。"""
    return -1.0 if 0 in events else 0.0

def boss_death_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """小怪/Boss 死亡奖励。"""
    return 1.0 if 1 in events else 0.0

def player_health_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """自身血量变化奖励。"""
    if 2 in events: # 掉血
        diff = prev_metrics['self_blood'] - next_metrics['self_blood']
        return -1 * diff * 0.05
    if 3 in events: # 回血
        return 10.0
    return 0.0

def boss_health_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """Boss 血量变化奖励。"""
    if 5 in events:
        diff = prev_metrics['boss_blood'] - next_metrics['boss_blood']
        return diff * 0.5
    return 0.0

def player_stamina_reward(env, prev_metrics, next_metrics, action, events, **kwargs):
    """自身架势/躯干奖励。"""
    if 6 in events:
        diff = next_metrics['self_stamina'] - prev_metrics['self_stamina']
        val = -3 * diff * 0.02
        if action == 2: # 格挡时不惩罚架势上升
            val = 0
        return val
    return 0.0

def boss_stamina_reward(env, prev_metrics, next_metrics, action, events, **kwargs):
    """Boss 架势/躯干奖励。"""
    if 7 in events:
        diff = next_metrics['boss_stamina'] - prev_metrics['boss_stamina']
        val = diff / 5
        if action == 5: # 忍具攻击
            val = max(5, val)
        if action in [2, 4]: # 格挡或闪避后的反击
            val = max(20, val)
        return val
    return 0.0

def survival_reward(env, **kwargs):
    """生存奖励（鼓励活下去）。"""
    return 0.1
