"""
模块用途：原子奖励函数定义（Reward Terms）。
遵循 Isaac Lab 风格，每个函数只负责一个维度的奖励计算。
"""

def player_death_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """自身死亡惩罚。返回 1.0，配合配置中的负权重实现惩罚。"""
    return 1.0 if 0 in events else 0.0

def boss_death_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """小怪/Boss 死亡奖励。返回 1.0，配合配置中的正权重实现奖励。"""
    return 1.0 if 1 in events else 0.0

def player_health_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """自身血量变化奖励。"""
    if 2 in events: # 掉血
        prev_val = prev_metrics.get('telemetry', {}).get('self_blood', 0)
        next_val = next_metrics.get('telemetry', {}).get('self_blood', 0)
        diff = prev_val - next_val
        return -1 * diff * 0.5
    return 0.0

def boss_health_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """Boss 血量变化奖励。"""
    if 5 in events:
        prev_val = prev_metrics.get('telemetry', {}).get('boss_blood', 0)
        next_val = next_metrics.get('telemetry', {}).get('boss_blood', 0)
        diff = prev_val - next_val
        if diff > 1000: diff = 0
        return diff * 0.5
    return 0.0

def player_stamina_reward(env, prev_metrics, next_metrics, action, events, **kwargs):
    """自身架势/躯干奖励。"""
    if 6 in events:
        prev_val = prev_metrics.get('telemetry', {}).get('self_stamina', 0)
        next_val = next_metrics.get('telemetry', {}).get('self_stamina', 0)
        # 数值减小代表架势条变长（恶化），diff 为正值
        diff = prev_val - next_val
        val = -3 * diff * 0.02
        if action == 2: # 格挡时不惩罚架势上升（鼓励格挡）
            val = 5
        return val
    return 0.0

def boss_stamina_reward(env, prev_metrics, next_metrics, action, events, **kwargs):
    """Boss 架势/躯干奖励。"""
    reward = 0
    
    # 1. 瞬时奖励：架势条正在减少（被攻击或被格挡）
    if 7 in events:
        prev_val = prev_metrics.get('telemetry', {}).get('boss_stamina', 0)
        next_val = next_metrics.get('telemetry', {}).get('boss_stamina', 0)
        diff = prev_val - next_val
        # 根据架势条变化值给分
        reward += diff * 0.1 # 缩小比例，使奖励更平滑
        
        # 针对特定动作的动作修正
        if action == 5: # 忍具攻击
            reward = min(2.0, reward)
        if action in [2]: # 格挡
            reward = max(5.0, reward)
            
    # 2. 持续压制奖励：仅当 Boss 架势条极低（快要破防）时才给分，且分值降低
    if 9 in events:
        reward += 0.5 # 只要压制住，每一步都给 0.5 分
        
    return reward

def survival_reward(env, **kwargs):
    """生存惩罚。"""
    return -0.1
