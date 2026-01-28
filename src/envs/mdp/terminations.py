"""
模块用途：定义环境终止条件。
"""

def check_simple_termination(events):
    """
    判断是否达到基础终止条件（如死亡、胜利）。
    - 0: 自身死亡
    - 1: Boss 死亡
    """
    if 0 in events or 1 in events:
        return True
    return False
