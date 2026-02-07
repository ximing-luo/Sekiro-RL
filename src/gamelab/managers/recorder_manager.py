import json
import os
import time
from typing import Dict, Any

class RecorderManager:
    """
    记录管理器：实现基于术语（Term-based）的数据记录。
    对标 Isaac Lab 的 RecorderManager。
    """
    def __init__(self, cfg: Dict[str, Any], save_dir="data/demos"):
        self.cfg = cfg
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.current_trajectory = []

    def record_step(self, env, obs, action, reward, next_obs, info):
        """遍历配置中的所有记录项。"""
        step_data = {"timestamp": time.time()}
        
        for name, term_cfg in self.cfg.items():
            val = term_cfg.func(
                env=env,
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
