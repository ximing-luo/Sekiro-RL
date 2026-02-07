from __future__ import annotations
import torch
import threading
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase
from .manager_term_cfg import ActionTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class ActionManager(ManagerBase):
    """动作管理器：实现基于术语的动作执行。
    
    继承自 ManagerBase，支持多动作项分发。
    """
    def __init__(self, cfg: Dict[str, ActionTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._term_names = list(self.cfg.keys())

    @property
    def active_terms(self) -> List[str]:
        return self._term_names

    def _prepare_terms(self):
        pass

    def apply_action(self, action: torch.Tensor):
        """执行动作。
        
        如果 action 是 tensor，则根据配置将其分发给对应的 Term。
        目前假设 action 的第一维是环境数量，第二维是动作空间。
        """
        # 简单起见，目前仍优先处理第一个 Term
        if self._term_names:
            term_name = self._term_names[0]
            term_cfg = self.cfg[term_name]
            
            # 获取动作内容
            # 如果是 batched action [num_envs, action_dim]，取第一个环境
            # 如果是单环境 action [action_dim]，直接使用
            if action.ndim == 2:
                action_to_apply = action[0]
            else:
                action_to_apply = action
            
            # 确保 action_to_apply 是可迭代的（对于 MultiDiscrete）
            # 如果是 0-d tensor，转换为 1-d
            if action_to_apply.ndim == 0:
                action_to_apply = action_to_apply.unsqueeze(0)
            
            # 获取动作函数
            fn = term_cfg.func(action_to_apply, **term_cfg.params)
            
            # 异步执行按键模拟
            if callable(fn):
                threading.Thread(target=fn, daemon=True).start()

    def reset(self, env_ids: Sequence[int] | None = None):
        return {}

    @property
    def action_term_dim(self) -> List[int]:
        """返回动作项的维度列表。"""
        dims = self.get_action_dim()
        if isinstance(dims, int):
            return [dims]
        return list(dims)

    def get_action_dim(self) -> int | List[int]:
        """获取动作维度。"""
        if self._term_names:
            term_cfg = self.cfg[self._term_names[0]]
            if 'dims' in term_cfg.params:
                return term_cfg.params['dims']
            return term_cfg.params.get('dim', 0)
        return 0
