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
    def __call__(self) -> torch.Tensor:
        """采样指令。"""
        raise NotImplementedError

class StandardCommandTerm(CommandTerm):
    """标准指令术语（包装旧的函数式逻辑）。"""
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(self) -> torch.Tensor:
        # 追求零成本抽象：假定底层函数返回符合契约的 Tensor
        return self.cfg.func(**self.cfg.params)

class CommandManager(ManagerBase):
    """指令管理器：生成并管理高层训练目标。
    """
    _TERM_CLASS = StandardCommandTerm

    def __init__(self, cfg: Dict[str, CommandTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._commands: Dict[str, torch.Tensor] = {}

    def step(self, dt: float = 0.0) -> Dict[str, torch.Tensor]:
        """计算/更新指令。"""
        # 简单逻辑：如果尚未生成指令，则生成
        if not self._commands:
            for name in self._term_names:
                term: CommandTerm = self._terms[name]
                self._commands[name] = term()
        return self._commands

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置指令。"""
        self._commands.clear()
        super().reset(env_ids)
        return self.step()

    def get_command(self, name: str) -> torch.Tensor | None:
        return self._commands.get(name)
