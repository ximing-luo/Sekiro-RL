from __future__ import annotations
import torch
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import CommandTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class CommandTerm(ManagerTermBase):
    """指令术语基类。"""
    @abstractmethod
    def __call__(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """采样指令。"""
        raise NotImplementedError

class StandardCommandTerm(CommandTerm):
    """标准指令术语（包装旧的函数式逻辑）。"""
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        # 追求零成本抽象：假定底层函数返回符合契约的 Tensor
        # 底层函数需支持处理特定的 env_ids
        if env_ids is None: env_ids = range(self._env.num_envs)
        return self.cfg.func(env_ids=env_ids, cfg=self.cfg)

class CommandManager(ManagerBase):
    """指令管理器：生成并管理高层训练目标。
    """
    _TERM_CLASS = StandardCommandTerm

    def __init__(self, cfg: Dict[str, CommandTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._commands: Dict[str, torch.Tensor] = {}
        # 初始化指令张量缓存
        for name, term in self._terms.items():
            # 预分配空间，确保并行性
            self._commands[name] = torch.zeros(
                (self.num_envs, *term.cfg.shape), device=self.device
            )

    def step(self, dt: float = 0.0) -> Dict[str, torch.Tensor]:
        """计算/更新指令。
        
        注意：在并行架构中，采样通常发生在 reset 时，step 仅负责返回或执行随时间变化的逻辑。
        """
        return self._commands

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置指令。仅针对指定的环境重新采样。"""
        if env_ids is None: env_ids = range(self.num_envs)
        
        for name in self._term_names:
            term: CommandTerm = self._terms[name]
            # 仅更新需要重置的环境
            self._commands[name][env_ids] = term(env_ids)
        
        super().reset(env_ids)
        return self._commands

    def get_command(self, name: str) -> torch.Tensor | None:
        return self._commands.get(name)
