"""
模块用途：原子奖励函数定义（Reward Terms）。
遵循 Isaac Lab 风格，每个函数只负责一个维度的奖励计算。
"""

def player_death_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """自身死亡惩罚。返回 10.0，配合配置中的正权重实现惩罚。"""
    return -10.0 if (0 in events) and (1 not in events) else 0.0

def boss_death_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """小怪/Boss 死亡奖励。返回 10.0，配合配置中的正权重实现奖励。"""
    return 10.0 if 1 in events else 0.0

def player_health_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """自身血量变化奖励。"""
    if 2 in events: # 掉血
        prev_val = prev_metrics.get('telemetry', {}).get('self_blood', 0)
        next_val = next_metrics.get('telemetry', {}).get('self_blood', 0)
        diff = prev_val - next_val
        # 纲量统一：100 变化对应 10.0 奖励
        return -1 * diff * 0.1
    return 0.0

def boss_health_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """Boss 血量奖励：包含血量变化奖励与低血量进度奖。"""
    reward = 0.0
    telemetry = next_metrics.get('telemetry', {})
    curr_hp = telemetry.get('boss_blood', 0)
    max_hp = telemetry.get('boss_blood_max', 1) # 避免除以 0
    
    # 2. 进度奖：Boss 血量越少，奖励越高 (微弱持续奖励)
    health_depletion_bonus = (1.0 - (curr_hp / max_hp)) * 0.1
    reward += health_depletion_bonus
    if health_depletion_bonus > 0.1:
        print(f"【异常】health_depletion_bonus: {health_depletion_bonus}")

    # 1. 瞬时掉血奖励 (基于差值)
    if 5 in events:
        prev_hp = prev_metrics.get('telemetry', {}).get('boss_blood', 0)
        hp_diff = prev_hp - curr_hp
        if 0 < hp_diff < 1000: # 过滤异常跳变
            reward += hp_diff * 0.08
            print(f"瞬时掉血奖励: reward: {reward}, 伤害: {hp_diff}, 引导分：{health_depletion_bonus}")
    return reward

def player_stamina_reward(env, prev_metrics, next_metrics, action, events, **kwargs):
    """自身架势/躯干奖励。"""
    if 8 in events:
        return -50 # 对应死半次的惩罚
    if 6 in events:
        prev_val = prev_metrics.get('telemetry', {}).get('self_stamina', 0)
        next_val = next_metrics.get('telemetry', {}).get('self_stamina', 0)
        # 数值减小代表架势条变长（恶化），diff 为正值
        diff = prev_val - next_val
        # 纲量统一：100 变化对应 1 惩罚
        val = -1 * diff * 0.01
    return 0.0

def boss_stamina_reward(env, prev_metrics, next_metrics, action, events, **kwargs):
    """Boss 架势/躯干奖励：包含架势恶化奖励、压制奖励及进度奖励。"""
    reward = 0.0
    telemetry = next_metrics.get('telemetry', {})
    curr_stamina = telemetry.get('boss_stamina', 0)
    max_stamina = telemetry.get('boss_stamina_max', 1)
    # 3. 进度奖：Boss 架势条越满（数值越低），奖励越高
    # 只狼中架势值减小代表进度增加
    posture_build_up_bonus = (1.0 - (curr_stamina / max_stamina)) * 0.1
    reward += posture_build_up_bonus
    if posture_build_up_bonus > 0.1:
        print(f"【异常】posture_build_up_bonus: {posture_build_up_bonus}")
    # 1. 瞬时奖励：架势条正在减少（被攻击或被格挡）
    if 7 in events:
        prev_stamina = prev_metrics.get('telemetry', {}).get('boss_stamina', 0)
        stamina_diff = prev_stamina - curr_stamina
        if 0 < stamina_diff < 1000: # 过滤异常跳变
            # 纲量统一：150 变化对应 12.0 奖励
            reward += stamina_diff * 0.08 
        if -1000 < stamina_diff < -1: # 过滤异常跳变
            reward += stamina_diff * 0.01
        print(f"架势奖励: reward: {reward}, 伤害: {stamina_diff}, 引导分: {posture_build_up_bonus}")
    return reward

def survival_reward(env, action, **kwargs):
    """生存奖励/惩罚。"""
    # 鼓励积极动作，对无意义动作（如 action=0 挂机）给微小惩罚，或者反过来
    reward = -0.3 if action not in [0,2] else 0.0
    reward -= 0.05 # 耗时惩罚
    return reward
