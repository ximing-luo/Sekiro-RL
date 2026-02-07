from typing import Dict
from src.gamelab.envs.manager_based_env_cfg import TerminationTermCfg

class TerminationManager:
    """
    终止管理器：实现基于术语的终止判定。
    """
    def __init__(self, cfg: Dict[str, TerminationTermCfg]):
        self.cfg = cfg

    def check_termination(self, env, prev_metrics, next_metrics, events):
        """遍历所有终止项，任何一项返回 True 则终止（或根据 Cfg 逻辑组合）。"""
        for name, term_cfg in self.cfg.items():
            if term_cfg.func(env=env, prev_metrics=prev_metrics, next_metrics=next_metrics, events=events, **term_cfg.params):
                return True
        return False

    def reset(self):
        pass
