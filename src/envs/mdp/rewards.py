"""
模块用途：原子奖励函数定义（Reward Terms）。
遵循 Isaac Lab 风格，每个函数只负责一个维度的奖励计算。
"""

import os
import re

def _log(message):
    """统一处理打印和文件记录"""
    print(message)
    _log_to_file(message)

def _log_to_file(message):
    """将消息写入根目录的 reward.txt，移除颜色代码"""
    clean_msg = re.sub(r'\033\[[0-9;]*m', '', message)
    log_path = os.path.join(os.getcwd(), "reward.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(clean_msg + "\n")

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
        # 纲量统一：100 伤害对应 1.0 原始分
        vla = -1 * diff * 0.01
        _log(f"\033[96m自身掉血惩罚: reward: {vla:.2f}, 伤害: {diff}\033[0m")
        return vla
    return 0.0

def boss_health_reward(env, prev_metrics, next_metrics, events, **kwargs):
    """Boss 血量奖励：包含血量变化奖励与低血量进度奖。"""
    reward = 0.0
    telemetry = next_metrics.get('telemetry', {})
    curr_hp = telemetry.get('boss_blood', 0)
    max_hp = telemetry.get('boss_blood_max', 1) # 避免除以 0
    
    # 2. 反引导：Boss 血量越少，惩罚越小
    health_depletion_bonus = - (curr_hp / max_hp) * 0.001
    reward += health_depletion_bonus

    # 1. 瞬时掉血奖励 (基于差值)
    if 5 in events:
        prev_hp = prev_metrics.get('telemetry', {}).get('boss_blood', 0)
        hp_diff = prev_hp - curr_hp
        if hp_diff > 0:
            # 纲量统一：100 伤害对应 1.0 原始分
            reward += hp_diff * 0.01
            _log(f"\033[93mBoss掉血奖励: reward: {reward:.2f}, 伤害: {hp_diff}, 引导分：{health_depletion_bonus:.4f}\033[0m")
        if -1000 < hp_diff < 0: # 回血惩罚
            reward += hp_diff * 0.001
    return reward

def player_stamina_reward(env, prev_metrics, next_metrics, action, events, **kwargs):
    """自身架势/躯干奖励。"""
    if 8 in events:
        return -10.0 # 与单次死亡对齐的惩罚
    if 6 in events:
        prev_val = prev_metrics.get('telemetry', {}).get('self_stamina', 0)
        next_val = next_metrics.get('telemetry', {}).get('self_stamina', 0)
        # 数值减小代表架势条变长（恶化），diff 为正值
        diff = prev_val - next_val
        # 纲量统一：100 变化对应 1.0 原始分
        if diff > 0: # 躯干上涨
            val = -1 * diff * 0.01
            _log(f"\033[96m自身架势惩罚: reward: {val:.2f}, 伤害: {diff}\033[0m")
            return val
        if -100 < diff < 0: # 躯干回跌
            val = diff * 0.001
            return val
    return 0.0

def boss_stamina_reward(env, prev_metrics, next_metrics, action, events, **kwargs):
    """Boss 架势/躯干奖励：包含架势恶化奖励、压制奖励及进度奖励。"""
    reward = 0.0
    telemetry = next_metrics.get('telemetry', {})
    curr_stamina = telemetry.get('boss_stamina', 0)
    max_stamina = telemetry.get('boss_stamina_max', 1)
    # 3. 进度奖：Boss 架势条越满（数值越低），奖励越高 (微弱持续奖励)
    # 只狼中架势值减小代表进度增加
    posture_build_up_bonus = (1.0 - (curr_stamina / max_stamina)) * 0.01
    reward += posture_build_up_bonus
    # 1. 瞬时奖励：架势条正在减少（被攻击或被格挡）
    if 7 in events:
        prev_stamina = prev_metrics.get('telemetry', {}).get('boss_stamina', 0)
        stamina_diff = prev_stamina - curr_stamina
        if 0 < stamina_diff < 1000: # 过滤血量空了卡bug
            # 纲量统一：100 伤害对应 1.0 原始分
            reward += stamina_diff * 0.01 
            _log(f"\033[93mBoss架势奖励: reward: {reward:.2f}, 伤害: {stamina_diff}, 引导分: {posture_build_up_bonus:.4f}\033[0m")
        if -1000 < stamina_diff < 0: # 回躯干惩罚
            reward += stamina_diff * 0.001
    return reward

def survival_reward(env, action, **kwargs):
    """生存奖励/惩罚：通过动作成本引导精准战斗。"""
    # 动作正则化：除观察(0)和防御(2)外，所有动作(攻击、闪避等)均有微小成本
    # 目的：减少无意义动作导致的后摇受击，强迫模型“看准再动”
    # 0.02 代表动作成本：若连续 50 帧乱按攻击，将亏损 1.0 原始分
    reward = -0.02 if action not in [0, 2] else 0.0
    reward += 0.01 if action in [6, 7, 8, 9] else 0.0
    # 纸人消耗 (action=5)：在动作成本基础上额外扣除
    reward -= 0.04 if action == 5 else 0.0
    
    # 耗时惩罚：促使对局进行的“底噪压力”
    # 0.01 代表时间成本：每过 100 帧(约 1.6 秒)扣除 1.0 原始分
    # 确保模型在安全观察的同时，仍有动力寻找进攻机会
    reward -= 0.01 
    return reward
