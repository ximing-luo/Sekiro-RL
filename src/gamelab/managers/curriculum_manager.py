from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase
from .manager_term_cfg import CurriculumTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class CurriculumManager(ManagerBase):
    """课程管理器：实现动态调整任务难度。
    
    对标 Isaac Lab，允许在重置或特定步数时更新环境参数。
    """
    def __init__(self, cfg: Dict[str, CurriculumTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._term_names = list(self.cfg.keys())

    @property
    def active_terms(self) -> List[str]:
        return self._term_names

    def _prepare_terms(self):
        pass

    def compute_curriculum(self, env_ids: Sequence[int] | None = None):
        """执行课程学习逻辑。
        
        通常在环境重置时调用，用于调整难度参数（如 Boss 属性）。
        """
        for name, term_cfg in self.cfg.items():
            # 调用课程函数
            term_cfg.func(env=self._env, env_ids=env_ids, **term_cfg.params)

    def reset(self, env_ids: Sequence[int] | None = None):
        # 课程逻辑通常在重置时被触发
        self.compute_curriculum(env_ids)
        return {}
