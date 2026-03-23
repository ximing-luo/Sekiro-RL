from __future__ import annotations
import torch
import os
import time
import numpy as np
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Any, List, Sequence
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RecorderTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class RecorderTerm(ManagerTermBase):
    """记录术语基类。
    
    定义了记录和导出的标准接口。
    """
    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._recording_enabled = [False] * self.num_envs

    def set_enabled(self, env_ids: Sequence[int], enabled: bool):
        """开关特定环境的记录。"""
        for i in env_ids:
            # 状态切换检测：如果从开启到关闭，触发一次导出
            if self._recording_enabled[i] and not enabled:
                self.export(env_ids=[i])
            self._recording_enabled[i] = enabled

    def __call__(self, obs: Any, action: Any, reward: Any, next_obs: Any, info: Any, env_ids: Sequence[int] | None = None) -> None:
        """执行术语逻辑，重定向到 record。"""
        return self.record(obs, action, reward, next_obs, info, env_ids)

    @abstractmethod
    def record(self, obs: Any, action: Any, reward: Any, next_obs: Any, info: Any, env_ids: Sequence[int] | None = None) -> None:
        """执行每一步的数据暂存逻辑。"""
        raise NotImplementedError

    @abstractmethod
    def export(self, env_ids: Sequence[int] | None = None) -> None:
        """执行轨迹结束后的数据持久化逻辑。"""
        raise NotImplementedError

class NpyRecorderTerm(RecorderTerm):
    """高性能 Numpy 数据记录术语。
    
    支持 .npy 格式存储，优化 I/O 与内存布局，为模仿学习与世界模型提供原始素材。
    """
    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg is not None:
            self.save_dir = "outputs/recordings"
            os.makedirs(self.save_dir, exist_ok=True)
        
        # 缓冲区：为每个环境独立分配
        self._obs_buffers: List[List[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self._action_buffers: List[List[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self._reward_buffers: List[List[float]] = [[] for _ in range(self.num_envs)]
        
        self._episode_counts = [0] * self.num_envs

    def record(self, obs: Dict[str, torch.Tensor], action: torch.Tensor, reward: torch.Tensor, next_obs: Dict[str, torch.Tensor], info: Dict[str, Any], env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = [i for i, enabled in enumerate(self._recording_enabled) if enabled]
            
        for i in env_ids:
            if not self._recording_enabled[i]:
                continue
            
            # 1. 记录观测 (优先记录 policy 图像)
            if "policy" in obs:
                # 已经是 (C, H, W) uint8 Tensor
                img = obs["policy"][i].detach().cpu().numpy()
                self._obs_buffers[i].append(img)
            else:
                img = obs[0][i].detach().cpu().numpy()
                self._obs_buffers[i].append(img)
            
            # 2. 记录动作 (支持异构动作空间，使用 int16 以兼容鼠标位移)
            act = action[i].detach().cpu().numpy().astype(np.int16)
            self._action_buffers[i].append(act)
            
            # 3. 记录奖励
            self._reward_buffers[i].append(float(reward[i].item()))
        
    def export(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = range(self.num_envs)
            
        for i in env_ids:
            if not self._obs_buffers[i]:
                continue
                
            timestamp = time.strftime("%m%d_%H%M%S")
            ep_dir = os.path.join(self.save_dir, f"{timestamp}_env_{i}_ep_{self._episode_counts[i]}")
            os.makedirs(ep_dir, exist_ok=True)
            
            # 转换为 numpy 数组并保存 (非压缩，支持 mmap)
            obs_arr = np.array(self._obs_buffers[i], dtype=np.uint8)
            act_arr = np.array(self._action_buffers[i], dtype=np.int16)
            rew_arr = np.array(self._reward_buffers[i], dtype=np.float32)
            
            np.save(os.path.join(ep_dir, "obs.npy"), obs_arr)
            np.save(os.path.join(ep_dir, "action.npy"), act_arr)
            np.save(os.path.join(ep_dir, "reward.npy"), rew_arr)
            
            print(f"[{self.__class__.__name__}] SekiroNpyData saved to {ep_dir} (frames: {len(obs_arr)})")
            
            # 重置缓冲区和计数
            self._obs_buffers[i] = []
            self._action_buffers[i] = []
            self._reward_buffers[i] = []
            self._episode_counts[i] += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """环境重置时，如果开启了录制，则导出当前轨迹。"""
        self.export(env_ids=env_ids)

class RecorderManager(ManagerBase):
    """记录管理器：实现基于术语的记录协调。
    
    对标 Isaac Lab 的 RecorderManager。
    """
    _TERM_CLASS = RecorderTerm
    def __init__(self, cfg: Dict[str, RecorderTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def set_recording_enabled(self, env_ids: Sequence[int] | int, enabled: bool):
        """动态开启/关闭录制。"""
        if isinstance(env_ids, int):
            env_ids = [env_ids]
            
        for term in self._terms.values():
            term.set_enabled(env_ids, enabled)

    def step(self, obs: Dict[str, torch.Tensor], action: torch.Tensor, reward: torch.Tensor, next_obs: Dict[str, torch.Tensor], info: Dict[str, Any]):
        """执行所有激活术语的记录逻辑。"""
        for term in self._terms.values():
            term(obs, action, reward, next_obs, info)

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置管理器及其所有术_语。"""
        super().reset(env_ids)
        return {}
