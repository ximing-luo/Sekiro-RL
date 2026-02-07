"""
通用原子奖励函数库。
这些函数不包含特定游戏的业务逻辑，而是处理通用的数据模式（如事件触发、数值变化、范围判定）。
"""

from typing import Any, Dict

def event_based_reward(env, prev_metrics: Dict, next_metrics: Dict, events: list, event_id: int, reward_val: float, **kwargs) -> float:
    """
    基于事件是否触发的奖励。
    Args:
        event_id: 监听的事件 ID。
        reward_val: 触发时返回的原始分。
    """
    return reward_val if event_id in events else 0.0

def attribute_delta_reward(env, prev_metrics: Dict, next_metrics: Dict, metric_key: str, sensor_name: str = "telemetry", scale: float = 1.0, **kwargs) -> float:
    """
    基于数值变化的奖励（例如血量减少、架势增加）。
    Args:
        metric_key: 要监听的指标键名（如 'self_blood'）。
        sensor_name: 传感器名称。
        scale: 缩放系数，用于统一纲量。
    """
    prev_val = prev_metrics.get(sensor_name, {}).get(metric_key, 0)
    next_val = next_metrics.get(sensor_name, {}).get(metric_key, 0)
    return (next_val - prev_val) * scale

def constant_reward(env, reward_val: float = 1.0, **kwargs) -> float:
    """
    常数奖励（如生存奖励）。
    """
    return reward_val

def value_mapping_reward(env, next_metrics: Dict, metric_key: str, sensor_name: str = "telemetry", func_type: str = "linear", scale: float = 1.0, **kwargs) -> float:
    """
    基于当前数值状态的奖励（如低血量惩罚、高进度奖励）。
    """
    val = next_metrics.get(sensor_name, {}).get(metric_key, 0)
    if func_type == "linear":
        return val * scale
    return 0.0
