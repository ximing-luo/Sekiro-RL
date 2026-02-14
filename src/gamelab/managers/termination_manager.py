from __future__ import annotations
import torch
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Any, Sequence, Tuple
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import TerminationTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class TerminationTerm(ManagerTermBase):
    """终止术语基类。"""
    @abstractmethod
    def __call__(self) -> torch.Tensor:
        """判定是否终止。"""
        raise NotImplementedError

class StandardTerminationTerm(TerminationTerm):
    """标准终止术语（包装旧的函数式逻辑）。"""
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._device_obj = torch.device(self.device)

    def __call__(self) -> torch.Tensor:
        # 直接调用函数，通过 env 引用自行获取所需数据
        return self.cfg.func(
            env=self._env, 
            **self.cfg.params
        )

class TerminationManager(ManagerBase):
    """终止管理器：实现基于术语的终止判定。
    """
    _TERM_CLASS = StandardTerminationTerm

    def __init__(self, cfg: Dict[str, TerminationTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算终止状态。对标 Isaac Lab。"""
        self._done_buf.zero_()
        self._time_out_buf.zero_()
        
        for name in self._term_names:
            term: TerminationTerm = self._terms[name]
            term_cfg = self.cfg[name]
            
            # 调用判定对象 (不再传递参数)
            res = term()
            
            # 更新缓存
            self._done_buf |= res
            if term_cfg.time_out:
                self._time_out_buf |= res
                
        return self._done_buf.clone(), self._time_out_buf.clone()

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置终止管理器。"""
        # 全部重置
        if env_ids is None:
            self._done_buf.zero_()
            self._time_out_buf.zero_()
        else:
            self._done_buf[env_ids] = False
            self._time_out_buf[env_ids] = False
            
        super().reset(env_ids)
        return {}
