from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Dict, List, Any, Sequence, Tuple
from .manager_base import ManagerBase
from .manager_term_cfg import TerminationTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class TerminationManager(ManagerBase):
    """终止管理器：实现基于术语的终止判定。
    
    对标 Isaac Lab，区分 done 和 time_out。
    """
    def __init__(self, cfg: Dict[str, TerminationTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._term_names = list(self.cfg.keys())
        self._done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def active_terms(self) -> List[str]:
        return self._term_names

    def _prepare_terms(self):
        pass

    def compute_terminations(self, prev_metrics: Any, next_metrics: Any, events: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算终止状态。
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (done, time_out)
        """
        self._done_buf.zero_()
        self._time_out_buf.zero_()
        
        for name, term_cfg in self.cfg.items():
            # 调用判定函数
            res = term_cfg.func(
                env=self._env, 
                prev_metrics=prev_metrics, 
                next_metrics=next_metrics, 
                events=events, 
                **term_cfg.params
            )
            
            # 确保是 tensor
            if not isinstance(res, torch.Tensor):
                res = torch.tensor(res, dtype=torch.bool, device=self.device).repeat(self.num_envs)
            
            # 更新缓存
            self._done_buf |= res
            if term_cfg.time_out:
                self._time_out_buf |= res
                
        return self._done_buf.clone(), self._time_out_buf.clone()

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._done_buf.zero_()
            self._time_out_buf.zero_()
        else:
            self._done_buf[env_ids] = False
            self._time_out_buf[env_ids] = False
        return {}
