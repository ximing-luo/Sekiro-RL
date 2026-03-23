from dataclasses import dataclass
from .manager_based_env_cfg import ManagerBasedEnvCfg

@dataclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    """基于管理器的强化学习环境配置。"""
    pass
