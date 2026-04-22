import numpy as np
from .base import BaseBuffer
from .batch import Batch
import configs.config as config

class ReplayBuffer(BaseBuffer):
    """
    经验回放缓冲区。
    采用预分配 NumPy 数组存储，确保物理内存连续性与类型安全。
    """
    def __init__(self, size, frame_history_len=4, obs_shape=None, alpha=None, beta=None):
        super().__init__(size)
        self.frame_history_len = frame_history_len
        self.obs_shape = obs_shape
        self.alpha = alpha if alpha is not None else config.cfg.per.alpha
        self.beta = beta if beta is not None else config.cfg.per.beta_start
        
        # 预分配存储空间
        self.observations = None  # 延迟分配，待第一次 add 时确定形状
        self.actions = np.zeros(size, dtype=np.int32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        self.priorities = np.zeros(size, dtype=np.float32)
        
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def add(self, obs, action, reward, done):
        """添加一条经验。"""
        # 第一次添加时初始化观测值数组
        if self.observations is None:
            obs_shape = obs.shape
            self.observations = np.zeros((self.capacity, *obs_shape), dtype=obs.dtype)
            self.obs_shape = obs_shape

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """采样并返回 Batch 对象。"""
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
        """批量获取堆叠后的观测值。"""
        obs = np.stack([self._get_stacked_obs(idx) for idx in indices])
        act = self.actions[indices]
        rew = self.rewards[indices]
        next_obs = np.stack([self._get_stacked_obs((idx + 1) % self.capacity) for idx in indices])
        done = self.dones[indices]
        return obs, act, rew, next_obs, done

    def _get_stacked_obs(self, idx):
        """
        获取以 idx 为结尾的堆叠观测。
        处理 Episode 边界与缓冲区初期的历史缺失问题。
        """
        frames = []
        for i in range(self.frame_history_len):
            # 计算回溯的索引
            curr_idx = (idx - i) % self.capacity
            
            is_out_of_buffer = False
            if self.size < self.capacity:
                if (idx - i) < 0:
                    is_out_of_buffer = True
            else:
                # 缓冲区已满，检查是否回溯到了当前写入位置 self.pos
                # 这里简化处理：如果在循环中遇到了 done，或者已经回溯了足够多的帧
                pass

            if is_out_of_buffer or (i > 0 and self.dones[curr_idx % self.capacity]):
                # 用当前最旧的一帧（即当前 episode 的起始帧）补齐
                fill_frame = frames[-1] if frames else self.observations[curr_idx]
                for _ in range(self.frame_history_len - len(frames)):
                    frames.append(fill_frame)
                break
            
            frames.append(self.observations[curr_idx])
            
        # 补齐因初始化阶段导致的缺失帧
        while len(frames) < self.frame_history_len:
            frames.append(frames[-1])
                
        return np.stack(list(reversed(frames)), axis=0)

    def update_priorities(self, indices, td_errors):
        """更新优先级 (PER)。"""
        td_errors = np.array(td_errors).flatten()
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = np.abs(error) + 1e-6
        self.max_priority = max(self.max_priority, np.max(np.abs(td_errors)))

    def __len__(self):
        return self.size
