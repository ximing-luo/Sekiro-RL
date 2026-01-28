from dataclasses import dataclass
from .manager_based_env_cfg import ManagerBasedEnvCfg

@dataclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    buffer_size: int = 50000
    frame_history_len: int = 4
    n_step_rewards: int = 20
