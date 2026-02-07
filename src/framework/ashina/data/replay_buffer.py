import numpy as np
import random
from collections import deque
from .base import BaseBuffer
from .batch import Batch
import configs.config as config

class ReplayBuffer(BaseBuffer):
    """
    支持 Batch 返回的经验回放缓冲区。
    """
    def __init__(self, size, frame_history_len=4, obs_shape=None, alpha=None, beta=None):
        super().__init__(size)
        self.frame_history_len = frame_history_len
        self.obs_shape = obs_shape
        self.alpha = alpha if alpha is not None else config.cfg.per.alpha
        self.beta = beta if beta is not None else config.cfg.per.beta_start
        
        # 简化存储
        self.observations = [None] * size
        self.actions = np.zeros(size, dtype=np.int32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        self.priorities = np.zeros(size, dtype=np.float32)
        
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def sample(self, batch_size: int) -> Batch:
        """采样并返回 Batch 对象"""
        if self.size == 0:
            return Batch()
        
        indices = np.random.choice(self.size, batch_size)
        
        obs, act, rew, obs_next, done = self._get_samples(indices)
        
        return Batch(
            obs=obs,
            act=act,
            rew=rew,
            obs_next=obs_next,
            done=done,
            indices=indices
        )

    def _get_samples(self, indices):
        """获取堆叠后的样本"""
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = [], [], [], [], []
        for idx in indices:
            # 这里简化了堆叠逻辑，实际应根据 frame_history_len 处理
            obs_batch.append(self.observations[idx])
            act_batch.append(self.actions[idx])
            rew_batch.append(self.rewards[idx])
            next_obs_batch.append(self.observations[(idx + 1) % self.capacity])
            done_batch.append(self.dones[idx])
            
        return (
            np.array(obs_batch),
            np.array(act_batch),
            np.array(rew_batch),
            np.array(next_obs_batch),
            np.array(done_batch)
        )

    def add(self, obs, action, reward, done):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def _get_samples(self, indices):
        obs = np.stack([self._get_stacked_obs(idx) for idx in indices])
        act = self.actions[indices]
        rew = self.rewards[indices]
        next_obs = np.stack([self._get_stacked_obs((idx + 1) % self.capacity) for idx in indices])
        done = self.dones[indices]
        return obs, act, rew, next_obs, done

    def _get_stacked_obs(self, idx):
        """获取以 idx 为结尾的堆叠观测。"""
        frames = []
        for i in range(self.frame_history_len):
            curr_idx = (idx - i) % self.capacity
            # 如果跨越了 episode 边界，则不再向前回溯，而是重复当前帧
            if i > 0 and self.dones[(curr_idx) % self.capacity]:
                # 补齐剩余帧
                for _ in range(self.frame_history_len - i):
                    frames.append(self.observations[(curr_idx + 1) % self.capacity])
                break
            frames.append(self.observations[curr_idx])
        
        if len(frames) < self.frame_history_len:
            # 补齐（以防万一）
            for _ in range(self.frame_history_len - len(frames)):
                frames.append(frames[-1])
                
        return np.stack(list(reversed(frames)), axis=0)

    def update_priorities(self, indices, td_errors):
        # 确保 td_errors 是 numpy 数组且有一致的形状
        td_errors = np.array(td_errors).flatten()
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = np.abs(error) + 1e-6
        self.max_priority = max(self.max_priority, np.max(np.abs(td_errors)))

    def can_sample(self, batch_size):
        return self.size >= max(batch_size, self.frame_history_len)

    def can_sample_n_step(self, batch_size, n):
        return self.size >= max(batch_size, self.frame_history_len + n)
