from __future__ import annotations
import torch
from gymnasium import spaces
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ActionTermCfg, MultiDiscreteActionTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class ActionTerm(ManagerTermBase):
    """动作术语基类。"""
    @property
    @abstractmethod
    def action_dim(self) -> int | List[int]:
        """返回该术语占用的动作维度。"""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, action: torch.Tensor) -> None:
        """应用动作。"""
        raise NotImplementedError

class MultiDiscreteActionTerm(ActionTerm):
    """多维离散动作术语。"""
    def __init__(self, cfg: MultiDiscreteActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.action_maps = cfg.action_maps
        self._dims = [len(m) for m in self.action_maps]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """重置动作术语。"""
        pass

    @property
    def action_dim(self) -> List[int]:
        return self._dims

    def __call__(self, action: torch.Tensor):
        """分发动作。
        
        基于“逻辑-效能同构”原则：动作函数本身已是非阻塞的（通过 GhostScheduler），
        因此直接在主线程调用即可，消除线程切换和队列开销。
        """
        actions_np = action.detach().cpu().numpy().astype(int)
        
        for env_id in range(self.num_envs):
            env_action = actions_np[env_id]
            for i, idx in enumerate(env_action):
                fn = self.action_maps[i].get(idx)
                if fn: fn()

class ActionManager(ManagerBase):
    """动作管理器：实现基于术语的动作执行。
    
    继承自 ManagerBase，支持多动作项分发。
    """
    __slots__ = ["_term_dispatch_list", "action"]

    def __init__(self, cfg: Dict[str, ActionTermCfg], env: ManagerBasedEnv):
        self._term_dispatch_list = []
        self.action: torch.Tensor = None
        super().__init__(cfg, env)

    def _prepare_terms(self):
        """实例化动作术语对象并预计算切片索引，物理化执行路径。"""
        start_idx = 0
        
        for name, term_cfg in self.cfg.items():
            term = self._instantiate_term(term_cfg, MultiDiscreteActionTerm)
            self._terms[name] = term
            self._term_names.append(name)
            
            # 物理化切片逻辑
            dim = term.action_dim
            width = len(dim) if isinstance(dim, list) else dim
            end_idx = start_idx + width
            
            # 预绑定分发闭包，消除 step 中的运行时计算
            def dispatch_fn(action, s=start_idx, e=end_idx, t=term):
                return t(action[:, s:e])
            
            self._term_dispatch_list.append(dispatch_fn)
            start_idx = end_idx

    def step(self, action: torch.Tensor):
        """执行动作。
        
        基于预绑定的分发列表，实现执行必然性。
        """
        self.action = action
        for dispatch in self._term_dispatch_list:
            dispatch(action)

    @property
    def action_term_dim(self) -> List[int]:
        """返回所有动作项的总维度列表。"""
        dims = []
        for name in self._term_names:
            term: ActionTerm = self._terms[name]
            dim = term.action_dim
            if isinstance(dim, list):
                dims.extend(dim)
            else:
                dims.append(dim)
        return dims

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """物理化指代动作空间。"""
        return spaces.MultiDiscrete(self.action_term_dim)
