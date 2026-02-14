"""
通用原子奖励函数库。
这些函数不包含特定游戏的业务逻辑，而是处理通用的数据模式（如事件触发、数值变化、范围判定）。
"""

from typing import Any

def constant_reward(env, reward_val: float = 1.0, **kwargs) -> float:
    """
    常数奖励（如生存奖励）。
    """
    return reward_val
