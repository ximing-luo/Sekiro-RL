from __future__ import annotations
import torch
from gymnasium import spaces
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ActionTermCfg, MultiDiscreteActionTermCfg, MultiBinaryActionTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class ActionTerm(ManagerTermBase):
    """动作术语基类。"""
    @property
    @abstractmethod
    def action_dim(self) -> List[int]:
        """返回该术语占用的动作维度列表。"""
        raise NotImplementedError

    def __call__(self, env_ids: Sequence[int] | None = None, action: torch.Tensor = None) -> None:
        """应用动作的入口。处理公共逻辑后分发给具体实现。"""
        if action is None: return
        if env_ids is None: env_ids = range(self.num_envs)
        
        self._apply_action(env_ids, action)

    @abstractmethod
    def _apply_action(self, env_ids: Sequence[int], action: torch.Tensor) -> None:
        """由子类实现的具体动作逻辑。"""
        raise NotImplementedError

class MultiDiscreteActionTerm(ActionTerm):
    """多维离散动作术语。"""
    def __init__(self, cfg: MultiDiscreteActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.action_maps = cfg.action_maps
        self._dims = [len(m) for m in self.action_maps]

    @property
    def action_dim(self) -> List[int]:
        return self._dims

    def _apply_action(self, env_ids: Sequence[int], action: torch.Tensor):
        """分发动作。实现执行必然性，减少运行时分支开销。"""
        # 1. 转换为 CPU NumPy 以便快速索引
        actions_np = action.detach().cpu().numpy().astype(int)
        
        # 2. 物理化分发：外层循环维度，内层循环环境，符合 CPU 缓存友好性
        for dim_idx, mapping in enumerate(self.action_maps):
            dim_actions = actions_np[:, dim_idx]
            for env_id in env_ids:
                # 利用 dict.get 消除显式的 if idx in mapping 判定
                fn = mapping.get(dim_actions[env_id])
                if fn: fn()

class MultiBinaryActionTerm(ActionTerm):
    """多维二值动作术语。"""
    def __init__(self, cfg: MultiBinaryActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.action_maps = cfg.action_maps
        self._n = cfg.n or len(self.action_maps)

    @property
    def action_dim(self) -> List[int]:
        # 对于二值动作，每个维度只有 2 个选项 (0/1)
        return [2] * self._n

    def _apply_action(self, env_ids: Sequence[int], action: torch.Tensor):
        """分发二值动作。每一位对应一个动作函数。"""
        # 1. 转换为 CPU NumPy
        actions_np = action.detach().cpu().numpy() > 0.5
        
        # 2. 物理化分发
        for dim_idx, fn in enumerate(self.action_maps):
            dim_actions = actions_np[:, dim_idx]
            for env_id in env_ids:
                if dim_actions[env_id]:
                    fn()

class NullActionTerm(ActionTerm):
    """不执行任何动作的术语，用于占位或记录。"""
    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._n = cfg.n or 1

    @property
    def action_dim(self) -> List[int]:
        # 返回 n 个连续的维度 (256 挡位作为示例)
        return [256] * self._n

    def _apply_action(self, env_ids: Sequence[int], action: torch.Tensor):
        # 仅占位，不执行任何驱动逻辑
        pass

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
            # 这里的 default_term_class 设为 MultiDiscreteActionTerm
            # 但如果 term_cfg.class_type 有值，则会使用配置的值
            term = self._instantiate_term(term_cfg, MultiDiscreteActionTerm)
            self._terms[name] = term
            self._term_names.append(name)
            
            # 物理化切片逻辑
            dim = term.action_dim
            width = len(dim)
            end_idx = start_idx + width
            
            # 预绑定分发闭包，消除 step 中的运行时计算
            def dispatch_fn(action, s=start_idx, e=end_idx, t=term):
                # 默认作用于所有环境
                return t(env_ids=None, action=action[:, s:e])
            
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
        for term in self._terms.values():
            dims.extend(term.action_dim)
        return dims

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """物理化指代动作空间。"""
        return spaces.MultiDiscrete(self.action_term_dim)
