from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union
import torch

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class ManagerTermBase(ABC):
    """管理器术语的基类。
    
    支持类形式的术语实现，允许术语拥有自己的状态。
    """
    def __init__(self, cfg: Any, env: ManagerBasedEnv):
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass

class ManagerBase(ABC):
    """所有管理器的基类。
    
    负责解析配置并提供统一的接口。
    """
    def __init__(self, cfg: Any, env: ManagerBasedEnv):
        self.cfg = cfg
        self._env = env
        self._term_names: List[str] = []
        
        if self.cfg:
            self._prepare_terms()

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    @abstractmethod
    def active_terms(self) -> List[str] | Dict[str, List[str]]:
        """返回当前激活的术语名称。"""
        raise NotImplementedError

    @abstractmethod
    def _prepare_terms(self):
        """从配置对象中准备术语信息。"""
        raise NotImplementedError

    def reset(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """重置管理器。"""
        return {}
