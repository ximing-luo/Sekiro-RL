"""
模块用途：动作映射与维度定义。
"""
from src.interfaces.controls import sekiro_ops as _actions

# 动作索引到可调用函数的映射
ACTION_FUNC_MAP = {
    0: _actions.no_op,
    1: _actions.attack,
    2: _actions.defense,
    3: _actions.jump,
    4: _actions.dodge_forward,
    5: _actions.ninja_attack,
    6: _actions.go_left,
    7: _actions.go_right,
    8: _actions.go_forward,
    9: _actions.go_back,
}

def sekiro_discrete_action(action_index: int, **kwargs):
    """
    基础离散动作映射。
    """
    return ACTION_FUNC_MAP.get(int(action_index), _actions.no_op)

def action_count():
    return len(ACTION_FUNC_MAP)
