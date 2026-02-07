from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase
from .manager_term_cfg import EventTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class EventManager(ManagerBase):
    """事件管理器：负责根据状态变化检测发生的事件。
    
    继承自 ManagerBase，对标 Isaac Lab。
    """
    def __init__(self, cfg: Dict[str, EventTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._term_names = list(self.cfg.keys())

    @property
    def active_terms(self) -> List[str]:
        return self._term_names

    def _prepare_terms(self):
        pass

    def compute_events(self, prev_metrics: Any, next_metrics: Any) -> List[Any]:
        """遍历配置中的所有事件项并执行。"""
        triggered_events = []
        for name, term_cfg in self.cfg.items():
            result = term_cfg.func(
                env=self._env,
                prev_metrics=prev_metrics,
                next_metrics=next_metrics,
                **term_cfg.params
            )
            
            if isinstance(result, bool):
                if result:
                    event_id = term_cfg.params.get("event_id", name)
                    triggered_events.append(event_id)
            elif result is not None:
                if isinstance(result, list):
                    triggered_events.extend(result)
                else:
                    triggered_events.append(result)
                    
        return triggered_events

    def reset(self, env_ids: Sequence[int] | None = None):
        return {}
