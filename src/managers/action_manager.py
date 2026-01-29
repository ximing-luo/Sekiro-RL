import threading
from typing import Dict
from src.envs.manager_based_env_cfg import ActionTermCfg

class ActionManager:
    """
    动作管理器：实现基于术语的动作执行。
    """
    def __init__(self, cfg: Dict[str, ActionTermCfg]):
        self.cfg = cfg
        # 目前 Sekiro 主要是离散动作空间，从第一个 Term 获取映射
        # 未来可以支持多个 Action Term 组合
        self.action_term = list(cfg.values())[0] if cfg else None

    def apply_action(self, env, action):
        """执行动作。"""
        if self.action_term:
            # 获取动作函数并异步执行
            fn = self.action_term.func(action, **self.action_term.params)
            threading.Thread(target=fn, daemon=True).start()

    def get_action_dim(self):
        # 假设 action_term 的 params 中包含 dim 信息
        return self.action_term.params.get('dim', 0) if self.action_term else 0
