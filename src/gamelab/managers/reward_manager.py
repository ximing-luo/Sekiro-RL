from __future__ import annotations
import torch
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Tuple
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class RewardTerm(ManagerTermBase):
    """奖励术语基类。"""
    @abstractmethod
    def __call__(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """计算奖励值。"""
        raise NotImplementedError

class StandardRewardTerm(RewardTerm):
    """标准奖励术语（包装旧的函数式逻辑）。"""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        # 追求零成本抽象：不再显式传递 metrics，让函数通过 env 自行获取
        val = self.cfg.func(env=self._env, env_ids=env_ids, cfg=self.cfg)
        return val * self.cfg.weight

class RewardManager(ManagerBase):
    """奖励管理器：实现基于术语（Term-based）的奖励计算。
    """
    _TERM_CLASS = StandardRewardTerm

    def __init__(self, cfg: Dict[str, RewardTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._reward_buf = torch.zeros(self.num_envs, device=self.device)

    def step(self, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算并返回总奖励及分量。对标 Isaac Lab。"""
        self._reward_buf.zero_()
        components = {}
        
        for name in self._term_names:
            term: RewardTerm = self._terms[name]
            
            # 计算原始值 (不再传递 metrics)
            val = term(env_ids=None)
                
            weighted_val = val * dt # 遵循 Isaac Lab 乘以 dt 的规范
            self._reward_buf += weighted_val
            
            components[name] = weighted_val
            
        return self._reward_buf.clone(), components
