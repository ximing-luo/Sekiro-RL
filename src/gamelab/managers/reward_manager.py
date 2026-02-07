from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase
from .manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class RewardManager(ManagerBase):
    """奖励管理器：实现基于术语（Term-based）的奖励计算。
    
    对标 Isaac Lab，支持 Tensor 化的奖励计算和 Episode 累加。
    """
    def __init__(self, cfg: Dict[str, RewardTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # 初始化缓存
        self._reward_buf = torch.zeros(self.num_envs, device=self.device)
        self._term_names = list(self.cfg.keys())
        
        # 记录每个 Term 的 Episode 累积奖励
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) 
            for name in self._term_names
        }

    @property
    def active_terms(self) -> List[str]:
        return self._term_names

    def _prepare_terms(self):
        # 可以在这里预处理 term 函数，检查合法性等
        pass

    def compute_reward(self, prev_metrics: Any, next_metrics: Any, action: Any, events: List[int]) -> torch.Tensor:
        """计算并返回总奖励。"""
        self._reward_buf.zero_()
        
        for name, term_cfg in self.cfg.items():
            # 计算奖励项原始值
            # 这里的 func 应该返回一个 (num_envs,) 的 tensor
            val = term_cfg.func(
                env=self._env,
                prev_metrics=prev_metrics,
                next_metrics=next_metrics,
                action=action,
                events=events,
                **term_cfg.params
            )
            
            # 确保是 tensor
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, device=self.device).repeat(self.num_envs)
                
            weighted_val = val * term_cfg.weight
            self._reward_buf += weighted_val
            
            # 更新累积值
            self._episode_sums[name] += weighted_val
            
        return self._reward_buf.clone()

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置指定环境的累积奖励。"""
        if env_ids is None:
            for buf in self._episode_sums.values():
                buf.zero_()
        else:
            for buf in self._episode_sums.values():
                buf[env_ids] = 0.0
        return {}

    def get_episode_sums(self) -> Dict[str, torch.Tensor]:
        """获取当前 Episode 的奖励统计。"""
        return self._episode_sums
