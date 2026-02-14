"""
只狼特定的事件检测。
"""

from typing import Dict, List, Any
import torch
from src.gamelab.managers.event_manager import EventTerm
from src.gamelab.managers.manager_term_cfg import EventTermCfg

class CompositeEventTerm(EventTerm):
    """复合事件术语。
    
    支持基于配置字典的批量事件检测。
    """
    def __call__(self) -> List[Any]:
        triggered = []
        event_configs = self.cfg.params.get("configs", {})
        
        # 直接从资产获取数据
        # 假设我们只关心 player 资产（因为 Sekiro 是单人游戏）
        asset = self._env.scene.assets.get("player")
        if asset is None:
            return triggered
            
        data = asset.data
        
        for eid, cfg in event_configs.items():
            etype = cfg.get("type")
            key = cfg.get("key")
            mode = cfg.get("mode", "changed")
            
            # 从 StateBuffer 自动获取当前值和前一时刻值
            if hasattr(data, key):
                nv = getattr(data, key)
                pv = getattr(data, f"prev_{key}")
            else:
                continue
            
            # 向量化检测：返回 (num_envs,) 的布尔 Tensor
            if etype == "delta":
                is_triggered = self._delta_event(pv, nv, mode)
            elif etype == "counter":
                is_triggered = self._counter_increment_event(pv, nv)
            elif etype == "threshold":
                is_triggered = self._threshold_event(nv, cfg.get("threshold", 0), mode)
            else:
                continue
                
            # 目前事件系统返回的是触发的 ID 列表。
            # 如果是向量化环境，这里需要特殊处理。
            if is_triggered.any():
                triggered.append(eid)
                
        return triggered

    def _threshold_event(self, next_val: torch.Tensor, threshold: float, mode: str = "below") -> torch.Tensor:
        if mode == "below": return next_val < threshold
        if mode == "above": return next_val > threshold
        return torch.zeros_like(next_val, dtype=torch.bool)

    def _delta_event(self, prev_val: torch.Tensor, next_val: torch.Tensor, mode: str = "changed") -> torch.Tensor:
        if mode == "increased": return next_val > prev_val
        if mode == "decreased": return next_val < prev_val
        if mode == "changed": return next_val != prev_val
        return torch.zeros_like(next_val, dtype=torch.bool)

    def _counter_increment_event(self, prev_count: torch.Tensor, next_count: torch.Tensor) -> torch.Tensor:
        return next_count > prev_count

def sekiro_events_cfg(configs: dict) -> EventTermCfg:
    """提供只狼事件检测的简洁配置。"""
    return EventTermCfg(
        class_type=CompositeEventTerm,
        params={"configs": configs}
    )

# 事件配置定义 (声明式)
SEKIRO_EVENT_CONFIGS = {
    0: {"type": "counter", "key": "player_deaths"},     # 自身死亡
    1: {"type": "counter", "key": "enemy_deaths"},      # Boss死亡
    2: {"type": "delta", "key": "player_hp", "mode": "decreased"}, # 自身掉血
    5: {"type": "delta", "key": "enemy_hp", "mode": "changed"},   # Boss血量变化
    6: {"type": "delta", "key": "player_posture", "mode": "changed"}, # 自身架势变化
    7: {"type": "delta", "key": "enemy_posture", "mode": "changed"}, # Boss架势变化
}
