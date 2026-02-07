"""
模块用途：原子终止条件定义（Termination Terms）。
"""

from src.gamelab.envs.mdp import time_out_termination

def player_dead_termination(env, events, **kwargs):
    """自身死亡导致回合结束。"""
    return 0 in events

def boss_dead_termination(env, events, **kwargs):
    """Boss 死亡导致回合结束。"""
    return 1 in events
