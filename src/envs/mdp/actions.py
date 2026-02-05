"""
模块用途：动作映射与维度定义。
"""
from src.interfaces.controls import sekiro_ops as _actions

# --- Multi-Discrete 动作头定义 ---
# 动作头 A (移动): [不动, 前, 后, 左, 右]
MOVE_ACTIONS = {
    0: _actions.no_op,
    1: _actions.go_forward,
    2: _actions.go_back,
    3: _actions.go_left,
    4: _actions.go_right,
}

# 动作头 B (动作/技能): [无, 攻击, 防御, 垫步, 跳跃]
SKILL_ACTIONS = {
    0: _actions.no_op,
    1: _actions.attack,
    2: _actions.defense,
    3: _actions.dodge_forward,
    4: _actions.jump,
}

MULTI_DISCRETE_DIMS = [len(MOVE_ACTIONS), len(SKILL_ACTIONS)]

def sekiro_multi_discrete_action(action_indices: list, **kwargs):
    """
    多维离散动作执行逻辑。
    将两个动作头的指令组合在一起并行执行。
    """
    move_idx, skill_idx = action_indices
    
    # 获取对应的原子动作函数
    move_fn = MOVE_ACTIONS.get(int(move_idx), _actions.no_op)
    skill_fn = SKILL_ACTIONS.get(int(skill_idx), _actions.no_op)
    
    def combined_action():
        # 这里使用简单的顺序触发，因为每个动作内部都有微小的 time.sleep
        # 由于是在独立的线程中运行，这种毫秒级的顺序触发在游戏里表现为“同时按下”
        move_fn()
        skill_fn()
        
    return combined_action

# 动作索引到可调用函数的映射 (保留用于兼容旧逻辑)
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

# 动作短标签，用于可视化展示
ACTION_LABELS = ['无操作', '攻击', '防御', '跳跃', '闪避', '忍义手', '左移', '右移', '前移', '后移']

# Multi-Discrete 标签定义
MULTI_DISCRETE_LABELS = [
    ['不动', '前移', '后移', '左移', '右移'], # 移动头
    ['无', '攻击', '防御', '垫步', '跳跃']   # 动作/技能头
]
MULTI_DISCRETE_HEAD_NAMES = ['Move', 'Skill']

# 无操作索引定义
no_op_index = 0

def sekiro_discrete_action(action_index: int, **kwargs):
    """
    基础离散动作映射。
    """
    return ACTION_FUNC_MAP.get(int(action_index), _actions.no_op)

def action_count():
    return len(ACTION_FUNC_MAP)
