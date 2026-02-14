from __future__ import annotations
import torch
from gymnasium import spaces
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Any
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ObservationTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class ObservationTerm(ManagerTermBase):
    """观测术语基类。"""
    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """返回该术语对应的观测空间。"""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """获取观测值。"""
        raise NotImplementedError

class StandardObservationTerm(ObservationTerm):
    """标准观测术语（包装旧的函数式逻辑）。"""
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # 优先使用显式声明，否则退化为物理化推导
        self._observation_space = self._derive_space()

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    def _derive_space(self) -> spaces.Box:
        """推导当前项的观测空间。"""
        # 严格遵循配置契约：直接根据元数据组装空间，不再有任何默认推测
        return spaces.Box(
            low=self.cfg.low,
            high=self.cfg.high,
            shape=self.cfg.shape,
            dtype=self.cfg.dtype
        )

    def __call__(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        # 直接调用函数，假定返回符合契约的 Tensor
        # 注意：观测项通常一次性计算所有环境，然后由 Manager 处理 view
        if env_ids is None: env_ids = range(self._env.num_envs)
        val = self.cfg.func(env=self._env, cfg=self.cfg)
        
        # 应用缩放（只有在需要时才应用）
        if self.cfg.scale != 1.0:
            val = val * self.cfg.scale
            
        return val

class ObservationManager(ManagerBase):
    """观测管理器：实现基于术语的观测提取。
    
    仅支持纯项观测（Dict），移除所有隐式拼接和分组逻辑。
    """ 
    _TERM_CLASS = StandardObservationTerm

    def __init__(self, cfg: Any, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # 观测数据缓存：存储最近一次计算的所有原始观测项
        self._obs_raw: Dict[str, torch.Tensor] = {}

    @property
    def raw(self) -> Dict[str, torch.Tensor]:
        """返回最近一次计算的所有原始观测数据（供其他管理器调用）。"""
        return self._obs_raw

    @property
    def observation_space(self) -> spaces.Dict:
        """返回所有观测项的空间字典。"""
        spaces_dict = {}
        for name, term in self._terms.items():
            spaces_dict[name] = term.observation_space
        return spaces.Dict(spaces_dict)

    def step(self) -> Dict[str, torch.Tensor]:
        """计算并更新所有观测项，并返回主观测张量。"""
        # 1. 计算并存入原始缓存 (供所有管理器共享)
        for name in self._term_names:
            term: ObservationTerm = self._terms[name]
            # 严格遵循配置契约：强制 view 为 (num_envs, *shape)
            self._obs_raw[name] = term().view(self.num_envs, *term.cfg.shape)
        
        return self._obs_raw