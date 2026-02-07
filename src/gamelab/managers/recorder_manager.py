from __future__ import annotations
import json
import os
import time
from typing import TYPE_CHECKING, Dict, Any, List
from .manager_base import ManagerBase

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class RecorderManager(ManagerBase):
    """
    记录管理器：实现基于术语（Term-based）的数据记录。
    对标 Isaac Lab 的 RecorderManager。
    """
    def __init__(self, cfg: Dict[str, Any], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.save_dir = "data/demos"
        os.makedirs(self.save_dir, exist_ok=True)
        self.current_trajectory = []

    @property
    def active_terms(self) -> List[str]:
        return list(self.cfg.keys())

    def _prepare_terms(self):
        pass

    def record_step(self, obs, action, reward, next_obs, info):
        """遍历配置中的所有记录项。"""
        step_data = {"timestamp": time.time()}
        
        for name, term_cfg in self.cfg.items():
            val = term_cfg.func(
                env=self._env,
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                info=info,
                **term_cfg.params
            )
            step_data[name] = val
            
        self.current_trajectory.append(step_data)

    def save_trajectory(self, filename=None):
        """保存当前轨迹。"""
        if not self.current_trajectory:
            return
        
        if filename is None:
            filename = f"demo_{int(time.time())}.json"
            
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.current_trajectory, f, indent=2)
            
        print(f"Trajectory saved to: {path}")
        self.current_trajectory = []
