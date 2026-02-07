"""
通用原子事件检测逻辑。
"""

from typing import Any, Dict, List

def threshold_event(next_val: float, threshold: float, mode: str = "below") -> bool:
    """基于阈值的事件检测。"""
    if mode == "below": return next_val < threshold
    if mode == "above": return next_val > threshold
    return False

def delta_event(prev_val: float, next_val: float, mode: str = "changed") -> bool:
    """基于变化的事件检测。"""
    if mode == "increased": return next_val > prev_val
    if mode == "decreased": return next_val < prev_val
    if mode == "changed": return next_val != prev_val
    return False

def counter_increment_event(prev_count: int, next_count: int) -> bool:
    """基于计数器增加的事件检测（如死亡次数）。"""
    return next_count > prev_count

def composite_events(env, prev_metrics: Dict, next_metrics: Dict, event_configs: Dict[int, Dict], sensor_name: str = "telemetry") -> List[int]:
    """
    基于配置的复合事件检测器。
    Args:
        event_configs: { event_id: { "type": "delta", "key": "hp", "mode": "decreased" } }
    """
    triggered = []
    prev_data = prev_metrics.get(sensor_name, {})
    next_data = next_metrics.get(sensor_name, {})
    
    if not prev_data or not next_data:
        return triggered

    for eid, cfg in event_configs.items():
        etype = cfg.get("type")
        key = cfg.get("key")
        mode = cfg.get("mode", "changed")
        
        pv = prev_data.get(key, 0)
        nv = next_data.get(key, 0)
        
        is_triggered = False
        if etype == "delta":
            is_triggered = delta_event(pv, nv, mode)
        elif etype == "counter":
            is_triggered = counter_increment_event(pv, nv)
        elif etype == "threshold":
            is_triggered = threshold_event(nv, cfg.get("threshold", 0), mode)
            
        if is_triggered:
            triggered.append(eid)
            
    return triggered
