"""
模块用途：原子终止条件定义（Termination Terms）。
"""

def player_dead_termination(env, events, **kwargs):
    """自身死亡导致回合结束。"""
    return 0 in events

def boss_dead_termination(env, events, **kwargs):
    """Boss 死亡导致回合结束。"""
    return 1 in events

def time_out_termination(env, **kwargs):
    """超时终止（如果环境中有步数限制）。"""
    # 暂时作为占位符，如果 env 有 max_episode_steps 可以增加逻辑
    return False
