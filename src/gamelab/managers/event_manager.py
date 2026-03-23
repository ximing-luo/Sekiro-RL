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
    def __call__(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """检测事件。必须返回触发的事件张量 [num_envs]。"""
        raise NotImplementedError

class StandardEventTerm(EventTerm):
    """标准事件术语（包装旧的函数式逻辑）。"""
    def __call__(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        # 契约：函数必须处理向量化逻辑并返回 Tensor
        return self.cfg.func(env=self._env, env_ids=env_ids, cfg=self.cfg)

class EventManager(ManagerBase):
    """事件管理器：实现基于术语的事件检测与处理。
    
    支持在环境重置或步进时触发特定的物理/状态变更。
    """
    _TERM_CLASS = StandardEventTerm

    def __init__(self, cfg: Dict[str, EventTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # 物理隔离：每个环境拥有独立的事件快照 [num_envs, num_terms]
        self.recent_events = torch.full((self.num_envs, len(self._term_names)), -1, dtype=torch.int16, device=self.device)

    def step(self, mode: str = "reset") -> torch.Tensor:
        """遍历所有事件项并检测。实现环境物理隔离。"""
        # 1. 清空快照 (-1 表示无事件)
        self.recent_events.fill_(-1)

        # 2. 聚合检测结果
        for i, name in enumerate(self._term_names):
            term_cfg = self.cfg[name]
            if term_cfg.mode != mode:
                continue
                
            term: EventTerm = self._terms[name]
            # 契约：__call__ 产出 (num_envs,)
            self.recent_events[:, i] = term(env_ids=None)
                    
        return self.recent_events
