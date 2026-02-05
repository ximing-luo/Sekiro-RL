from typing import Dict, Any
from src.envs.manager_based_env_cfg import ObservationTermCfg

class ObservationManager:
    """
    观测管理器：实现基于术语的观测提取。
    """
    def __init__(self, cfg: Dict[str, ObservationTermCfg]):
        self.cfg = cfg

    def compute_observations(self, env):
        """遍历配置中的所有观测项。"""
        observations = {}
        for name, term_cfg in self.cfg.items():
            val = term_cfg.func(env=env, **term_cfg.params)
            observations[name] = val
        return observations

    def reset(self):
        """重置管理器状态（如果有）。"""
        pass
