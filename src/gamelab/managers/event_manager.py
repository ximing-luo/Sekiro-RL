from __future__ import annotations
import torch
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import EventTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class EventTerm(ManagerTermBase):
    """事件术语基类。"""
    @abstractmethod
    def __call__(self) -> List[Any]:
        """检测事件。必须返回触发的事件列表。"""
        raise NotImplementedError

class StandardEventTerm(EventTerm):
    """标准事件术语（包装旧的函数式逻辑）。"""
    def __call__(self) -> List[Any]:
        # 执行逻辑 (不再传递 metrics)
        result = self.cfg.func(
            env=self._env,
            **self.cfg.params
        )
        
        # 归一化逻辑
        if result is None:
            return []
            
        if isinstance(result, bool):
            return [self.cfg.event_id] if result and self.cfg.event_id else []
            
        if isinstance(result, list):
            return result
            
        return [result]

class EventManager(ManagerBase):
    """事件管理器：实现基于术语的事件检测与处理。
    
    支持在环境重置或步进时触发特定的物理/状态变更。
    """
    _TERM_CLASS = StandardEventTerm

    def __init__(self, cfg: Dict[str, EventTermCfg], env: ManagerBasedEnv):
        self.recent_events = []
        super().__init__(cfg, env)

    def step(self, mode: str = "reset") -> List[Any]:
        """遍历所有事件项并检测。对标 Isaac Lab。"""
        triggered_events = []
        for name in self._term_names:
            term_cfg = self.cfg[name]
            if term_cfg.mode != mode:
                continue
                
            term: EventTerm = self._terms[name]
            events = term()
            if events:
                triggered_events.extend(events)
                    
        self.recent_events = triggered_events
        return triggered_events
