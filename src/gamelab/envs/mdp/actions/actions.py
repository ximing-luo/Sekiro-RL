"""
通用原子动作处理函数库。
"""

from typing import Any, Dict, List, Callable

def multi_discrete_action_provider(action_indices: List[int], action_maps: List[Dict[int, Callable]], **kwargs):
    """
    通用的多维离散动作映射执行器。
    Args:
        action_indices: 智能体输出的动作索引列表 [idx1, idx2, ...]
        action_maps: 每一维对应的动作函数映射字典列表
    """
    def combined_action():
        for i, idx in enumerate(action_indices):
            if i < len(action_maps):
                action_fn = action_maps[i].get(int(idx))
                if action_fn:
                    action_fn()
    return combined_action

def discrete_action_provider(action_index: int, action_map: Dict[int, Callable], **kwargs):
    """
    通用的单维离散动作映射执行器。
    """
    return action_map.get(int(action_index))
