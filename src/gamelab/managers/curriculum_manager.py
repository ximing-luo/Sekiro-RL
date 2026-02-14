from __future__ import annotations
import torch
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import CurriculumTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class CurriculumTerm(ManagerTermBase):
    """课程术语基类。"""
    @abstractmethod
    def __call__(self, env_ids: Sequence[int] | None = None) -> None:
        """执行课程逻辑。"""
        raise NotImplementedError

class StandardCurriculumTerm(CurriculumTerm):
    """标准课程术语（包装旧的函数式逻辑）。"""
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(self, env_ids: Sequence[int] | None = None) -> None:
        self.cfg.func(env=self._env, env_ids=env_ids, **self.cfg.params)

class CurriculumManager(ManagerBase):
    """课程管理器：实现动态任务难度调整。
    """
    _TERM_CLASS = StandardCurriculumTerm

    def __init__(self, cfg: Dict[str, CurriculumTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def step(self, env_ids: Sequence[int] | None = None):
        """执行课程学习逻辑。
        
        通常在环境重置时调用，用于调整难度参数（如 Boss 属性）。
        """
        for name in self._term_names:
            term: CurriculumTerm = self._terms[name]
            term(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置课程管理器。"""
        # 课程逻辑通常在重置时被触发
        self.step(env_ids)
        super().reset(env_ids)
        return {}
