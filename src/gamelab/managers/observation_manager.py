from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Dict, List, Any, Sequence, Union
from .manager_base import ManagerBase
from .manager_term_cfg import ObservationTermCfg, ObservationGroupCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class ObservationManager(ManagerBase):
    """观测管理器：实现基于术语的观测提取。
    
    支持观测组、自动拼接和 Tensor 缓存。
    """
    def __init__(self, cfg: Dict[str, ObservationGroupCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._group_obs: Dict[str, torch.Tensor] = {}
        self._term_values: Dict[str, Dict[str, torch.Tensor]] = {}

    @property
    def active_terms(self) -> Dict[str, List[str]]:
        return {group_name: list(group_cfg.terms.keys()) for group_name, group_cfg in self.cfg.items()}

    def _prepare_terms(self):
        # 初始化观测缓存
        for group_name, group_cfg in self.cfg.items():
            self._term_values[group_name] = {}

    def compute_observations(self) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """计算并返回所有观测组的观测值。"""
        results = {}
        
        for group_name, group_cfg in self.cfg.items():
            group_results = []
            
            for term_name, term_cfg in group_cfg.terms.items():
                # 调用观测函数
                val = term_cfg.func(env=self._env, **term_cfg.params)
                
                # 确保是 tensor 且在正确的设备上
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, device=self.device)
                
                # 应用缩放
                if term_cfg.scale != 1.0:
                    val = val * term_cfg.scale
                
                # 存储原始项值
                self._term_values[group_name][term_name] = val
                group_results.append(val)
            
            # 处理组拼接
            if group_cfg.concatenate_terms:
                # 假设所有项在最后一维拼接
                results[group_name] = torch.cat(group_results, dim=-1)
            else:
                results[group_name] = self._term_values[group_name]
                
        return results

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置管理器状态。"""
        return {}
