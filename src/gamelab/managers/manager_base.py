from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv
    from src.gamelab.managers.manager_term_cfg import ManagerTermBaseCfg

class ManagerTermBase(ABC):
    """管理器术语的基类。
    
    所有术语必须实现 __call__ 和 reset 接口，确保契约的神圣性。
    """
    __slots__ = ["cfg", "_env"]

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """重置术语状态。子类可按需实现。"""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """执行术语逻辑。"""
        raise NotImplementedError

    def __repr__(self) -> str:
        """返回术语的字符串表示。"""
        return f"{self.__class__.__name__}(cfg={self.cfg})"

class ManagerBase(ABC):
    """所有管理器的基类。
    
    负责解析配置并提供统一的接口。
    """
    __slots__ = ["cfg", "_env", "_term_names", "_terms"]

    # 默认术语类，子类可覆盖
    _TERM_CLASS: type[ManagerTermBase] | None = None

    def __init__(self, cfg: Any, env: ManagerBasedEnv):
        self.cfg = cfg
        self._env = env
        self._term_names: List[str] = []
        self._terms: Dict[str, ManagerTermBase] = {}
        
        if self.cfg:
            self._prepare_terms()

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def active_terms(self) -> List[str]:
        """返回当前激活的术语名称。"""
        return self._term_names

    def _prepare_terms(self):
        """从配置对象中准备术语信息。"""
        if self._TERM_CLASS is None: return

        for name, term_cfg in self.cfg.items():
            if term_cfg is not None:
                term = self._instantiate_term(term_cfg, self._TERM_CLASS)
                self._terms[name] = term
        self._term_names = list(self._terms.keys())

    @abstractmethod
    def step(self, *args, **kwargs) -> Any:
        """执行管理器的步进逻辑。"""
        raise NotImplementedError

    def _instantiate_term(self, cfg: type[ManagerTermBaseCfg], default_term_class: type[ManagerTermBase]) -> ManagerTermBase:
        """根据配置实例化术语对象。"""
        cls = cfg.class_type or default_term_class
        return cls(cfg, self._env)

    def reset(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        """重置管理器及其所有术语。"""
        # 展平逻辑，消除递归熵增。子类如 ObservationManager 应自行管理嵌套术语的重置。
        for term in self._terms.values():
            if isinstance(term, ManagerTermBase):
                term.reset(env_ids)
        return {}
