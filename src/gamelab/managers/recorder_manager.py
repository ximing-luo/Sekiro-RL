from __future__ import annotations
import json
import os
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Any, List, Sequence
from .manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class RecorderTerm(ManagerTermBase):
    """记录术语基类。"""
    @abstractmethod
    def __call__(self, obs: Any, action: Any, reward: Any, next_obs: Any, info: Any) -> Any:
        """执行记录逻辑。"""
        raise NotImplementedError

class StandardRecorderTerm(RecorderTerm):
    """标准记录术语（包装旧的函数式逻辑）。"""
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(self, obs: Any, action: Any, reward: Any, next_obs: Any, info: Any) -> Any:
        return self.cfg.func(
            env=self._env,
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            info=info,
            **self.cfg.params
        )

class RecorderManager(ManagerBase):
    """
    记录管理器：实现基于术语（Term-based）的数据记录。
    对标 Isaac Lab 的 RecorderManager。
    """
    _TERM_CLASS = StandardRecorderTerm

    def __init__(self, cfg: Dict[str, Any], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.save_dir = "outputs/data/demos"
        os.makedirs(self.save_dir, exist_ok=True)
        self.current_trajectory = []

    def step(self, obs, action, reward, next_obs, info):
        """遍历配置中的所有记录项。"""
        step_data = {"timestamp": time.time()}
        
        for name in self._term_names:
            term: RecorderTerm = self._terms[name]
            val = term(obs, action, reward, next_obs, info)
            step_data[name] = val
            
        self.current_trajectory.append(step_data)

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置记录管理器。"""
        super().reset(env_ids)
        return {}

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
