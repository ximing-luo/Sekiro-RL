"""
模块用途：提供动作索引到可调用函数的映射与展示标签。

包含：
- 常量：ACTION_FUNC_MAP, ACTION_LABELS, ACTION_DIM
- 函数：get_action_callable(index), action_count(), assert_config_consistency(expected_dim=None)

边界：
- 负责：将离散动作编号映射到执行函数
- 不负责：具体动作实现、环境状态管理、训练逻辑
"""
import os
import sys

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

# 动作索引到展示标签的映射
ACTION_LABELS = [
    '无动','攻击','格挡','跳跃','闪避','忍具','左移','右移','前进','后退'
]

a = 3
# 动作索引到触发尖峰的强度映射
ACTION_SPIKE_AMOUNTS = [
    a*0.0,  # 无动
    a*1.3,  # 攻击
    a*1.1,  # 格挡
    a*1.6,  # 跳跃
    a*1.2,  # 闪避
    a*2.5,  # 忍具
    a*1.0,  # 左移
    a*1.0,  # 右移
    a*1.2,  # 前进
    a*1.0,  # 后退
]

# 动作空间维度统一来源
ACTION_DIM = len(ACTION_FUNC_MAP)

def get_action_callable(index: int):
    return ACTION_FUNC_MAP.get(int(index), _actions.no_op)

def action_count():
    return ACTION_DIM

def assert_config_consistency(expected_dim: int | None = None):
    dim = expected_dim if expected_dim is not None else ACTION_DIM
    assert len(ACTION_FUNC_MAP) == dim, f"动作映射数量({len(ACTION_FUNC_MAP)})与期望维度({dim})不一致"

def no_op_index() -> int:
    for idx, fn in ACTION_FUNC_MAP.items():
        if fn is _actions.no_op:
            return int(idx)
    return 0

def get_action_spike_amounts():
    return list(ACTION_SPIKE_AMOUNTS)