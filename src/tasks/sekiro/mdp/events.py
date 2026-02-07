"""
只狼特定的事件检测。
"""

from src.gamelab.envs.mdp import composite_events

# 事件配置定义 (声明式)
SEKIRO_EVENT_CONFIGS = {
    0: {"type": "counter", "key": "player_deaths"},     # 自身死亡
    1: {"type": "counter", "key": "enemy_deaths"},      # Boss死亡
    2: {"type": "delta", "key": "self_blood", "mode": "decreased"}, # 自身掉血
    5: {"type": "delta", "key": "boss_blood", "mode": "changed"},   # Boss血量变化
    6: {"type": "delta", "key": "self_stamina", "mode": "changed"}, # 自身架势变化
    7: {"type": "delta", "key": "boss_stamina", "mode": "changed"}, # Boss架势变化
}

def sekiro_events_logic(env, prev_metrics, next_metrics, **kwargs):
    """
    只狼事件检测：直接调用通用复合检测器。
    """
    events = composite_events(
        env=env,
        prev_metrics=prev_metrics,
        next_metrics=next_metrics,
        event_configs=SEKIRO_EVENT_CONFIGS,
        **kwargs
    )
    
    # 可以在这里保留特有的复杂逻辑，如“架势崩溃”
    # ... 
    
    return events
