from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Dict, List, Any, Sequence
from .manager_base import ManagerBase
from .manager_term_cfg import CommandTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class CommandManager(ManagerBase):
    """指令管理器：负责采样和管理高层训练目标（Commands）。
    
    继承自 ManagerBase，对标 Isaac Lab。
    """
    def __init__(self, cfg: Dict[str, CommandTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._term_names = list(self.cfg.keys())
        self._commands: Dict[str, torch.Tensor] = {}

    @property
    def active_terms(self) -> List[str]:
        return self._term_names

    def _prepare_terms(self):
        pass

    def compute_commands(self) -> Dict[str, torch.Tensor]:
        """计算/更新指令。"""
        # 简单逻辑：如果尚未生成指令，则生成
        if not self._commands:
            for name, term_cfg in self.cfg.items():
                val = term_cfg.func(**term_cfg.params)
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, device=self.device)
                self._commands[name] = val
        return self._commands

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置指令。"""
        self._commands.clear()
        return self.compute_commands()

    def get_command(self, name: str) -> torch.Tensor | None:
        return self._commands.get(name)
