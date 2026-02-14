from __future__ import annotations
import torch
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
    def __call__(self, obs: Any, action: Any, reward: Any, next_obs: Any, info: Any, env_ids: Sequence[int] | None = None) -> Any:
        """执行记录逻辑。"""
        raise NotImplementedError

class StandardRecorderTerm(RecorderTerm):
    """标准记录术语（包装旧的函数式逻辑）。"""
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(self, obs: Any, action: Any, reward: Any, next_obs: Any, info: Any, env_ids: Sequence[int] | None = None) -> Any:
        return self.cfg.func(
            env=self._env,
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            info=info,
            env_ids=env_ids,
            cfg=self.cfg
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
        # 为每个环境独立分配缓冲区，实现多轨并行记录
        self._buffers: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_envs)]
        # 记录每个环境的轨迹计数，用于命名
        self._episode_counts = torch.zeros(self.num_envs, dtype=torch.int)

    def step(self, obs, action, reward, next_obs, info):
        """记录并行环境中的步进数据。"""
        # 计算所有 Term 的结果 (通常返回 Batch Tensor)
        step_results = {}
        for name in self._term_names:
            term: RecorderTerm = self._terms[name]
            step_results[name] = term(obs, action, reward, next_obs, info)

        # 核心：将 Batch 数据分发（Dispatch）到各环境的私有缓冲区
        for i in range(self.num_envs):
            data = {"timestamp": time.time()}
            for name, val in step_results.items():
                # 如果是 Tensor，提取对应环境的切片并转为 CPU 副本以节省显存
                if isinstance(val, torch.Tensor):
                    data[name] = val[i].detach().cpu()
                else:
                    data[name] = val[i] if isinstance(val, (list, tuple)) else val
            self._buffers[i].append(data)

    def reset(self, env_ids: Sequence[int] | None = None):
        """当环境重置时，自动保存并清理该环境的轨迹缓冲区。"""
        if env_ids is None:
            env_ids = range(self.num_envs)
        
        for i in env_ids:
            if len(self._buffers[i]) > 0:
                # 触发保存逻辑：将完整的一轮轨迹持久化
                self.save_trajectory(env_id=i)
                # 清空缓冲区，迎接下一轮
                self._buffers[i] = []
                self._episode_counts[i] += 1
                
        super().reset(env_ids)
        return {}

    def save_trajectory(self, env_id: int):
        """保存特定环境的轨迹。"""
        buffer = self._buffers[env_id]
        if not buffer:
            return
            
        ep_idx = self._episode_counts[env_id].item()
        filename = f"env_{env_id}_ep_{ep_idx}_{int(time.time())}.pt"
        path = os.path.join(self.save_dir, filename)
        
        # 使用 torch.save 替代 JSON，实现高性能二进制存储
        torch.save(buffer, path)
        print(f"Parallel Trajectory saved: {path} (length: {len(buffer)})")
