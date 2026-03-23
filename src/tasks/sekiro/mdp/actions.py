"""
只狼特定的动作定义。
通过配置通用执行器来实现，不再包含硬编码逻辑。
"""

from src.gamelab.assets.sekiro import ops as _actions
from src.gamelab.managers.action_manager import MultiDiscreteActionTerm, MultiBinaryActionTerm, NullActionTerm
from src.gamelab.managers.manager_term_cfg import ActionTermCfg, MultiDiscreteActionTermCfg, MultiBinaryActionTermCfg

def body_action_cfg(action_maps: list) -> MultiDiscreteActionTermCfg:
    """提供只狼身体动作的简洁配置。"""
    return MultiDiscreteActionTermCfg(
        class_type=MultiDiscreteActionTerm,
        action_maps=action_maps
    )

def sekiro_bitmask_action_cfg() -> MultiBinaryActionTermCfg:
    """提供 11 位掩码动作配置。"""
    return MultiBinaryActionTermCfg(
        class_type=MultiBinaryActionTerm,
        action_maps=BIT_MASK_MAP
    )

def sekiro_mouse_action_cfg() -> ActionTermCfg:
    """提供鼠标视角动作配置（仅用于记录）。"""
    return ActionTermCfg(
        class_type=NullActionTerm,
        n=2  # dx, dy
    )

# 动作映射定义 (纯数据)
# 11 位掩码映射 (去掉了离散视角按键)
BIT_MASK_MAP = [
    _actions.go_forward,   # 0: W
    _actions.go_left,      # 1: A
    _actions.go_back,      # 2: S
    _actions.go_right,     # 3: D
    _actions.jump,         # 4: SPACE
    _actions.attack,       # 5: J
    _actions.defense,      # 6: G
    _actions.F_go,         # 7: Q
    _actions.dodge_forward,# 8: SHIFT
    _actions.use_items,    # 9: R
    _actions.lock_vision,  # 10: X
]
MOVE_MAP = {
    0: _actions.no_op,
    1: _actions.go_forward,
    2: _actions.go_back,
    3: _actions.go_left,
    4: _actions.go_right,
}

SKILL_MAP = {
    0: _actions.no_op,
    1: _actions.defense
}

# 维度信息供配置类使用
MULTI_DISCRETE_DIMS = [len(MOVE_MAP), len(SKILL_MAP)]

# 标签定义 (UI/可视化用)
MULTI_DISCRETE_LABELS = [
    ['不动', '前移', '后移', '左移', '右移'],
    ['无', '防御', '垫步']
]
MULTI_DISCRETE_HEAD_NAMES = ['Move', 'Skill']
