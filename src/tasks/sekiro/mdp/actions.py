"""
只狼特定的动作定义。
通过配置通用执行器来实现，不再包含硬编码逻辑。
"""

from src.gamelab.assets.sekiro import ops as _actions
from src.gamelab.envs.mdp.actions import multi_discrete_action_provider

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

def sekiro_multi_discrete_action(action_indices: list, **kwargs):
    """
    只狼多维动作：直接调用通用映射器。
    """
    return multi_discrete_action_provider(
        action_indices=action_indices,
        action_maps=[MOVE_MAP, SKILL_MAP],
        **kwargs
    )

# 标签定义 (UI/可视化用)
MULTI_DISCRETE_LABELS = [
    ['不动', '前移', '后移', '左移', '右移'],
    ['无', '防御', '垫步']
]
MULTI_DISCRETE_HEAD_NAMES = ['Move', 'Skill']
