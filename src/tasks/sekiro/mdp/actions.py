"""
只狼特定的动作定义。
通过配置通用执行器来实现，不再包含硬编码逻辑。
"""

from src.gamelab.assets.sekiro import ops as _actions
from src.gamelab.managers.action_manager import MultiDiscreteActionTerm
from src.gamelab.managers.manager_term_cfg import ActionTermCfg, MultiDiscreteActionTermCfg

def body_action_cfg(action_maps: list) -> MultiDiscreteActionTermCfg:
    """提供只狼身体动作的简洁配置。"""
    return MultiDiscreteActionTermCfg(
        class_type=MultiDiscreteActionTerm,
        action_maps=action_maps
    )

# 动作映射定义 (纯数据)
MOVE_MAP = {
    0: _actions.no_op,
    1: _actions.go_forward,
    2: _actions.go_back,
    3: _actions.go_left,
    4: _actions.go_right,
}

SKILL_MAP = {
    0: _actions.no_op,
    1: _actions.defense,
    2: _actions.dodge_forward,
}

# 维度信息供配置类使用
MULTI_DISCRETE_DIMS = [len(MOVE_MAP), len(SKILL_MAP)]

# 标签定义 (UI/可视化用)
MULTI_DISCRETE_LABELS = [
    ['不动', '前移', '后移', '左移', '右移'],
    ['无', '防御', '垫步']
]
MULTI_DISCRETE_HEAD_NAMES = ['Move', 'Skill']
