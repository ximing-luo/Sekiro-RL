import numpy as np
import random
from collections import deque
from src.policies.buffers.base import BaseBuffer

class ReplayBuffer(BaseBuffer):
    """
    一个通用的经验回放缓冲区，支持：
    1. 帧堆叠 (Frame Stacking)
    2. 优先经验回放 (Prioritized Experience Replay)
    3. N-步奖励 (N-step Rewards)
    """
    def __init__(self, capacity, frame_history_len=4, alpha=0.6, beta=0.4):
        super().__init__(capacity)
        self.frame_history_len = frame_history_len
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = 0.001
        
        # 使用循环数组存储，提高效率
        self.observations = [None] * capacity
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        # 用于视频流同步的辅助变量 (保持与旧代码兼容)
        self.video_num_in_buffer = 0

    def add(self, obs, action, reward, done):
        """添加一条经验。注意：obs 应该是单帧。"""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.video_num_in_buffer = self.size

    def store_frame(self, frame):
        """兼容旧接口：存储单帧图像。"""
        self.observations[self.pos] = frame
        self.priorities[self.pos] = self.max_priority
        # 注意：此处不增加 size，因为还没有动作和奖励
        # 但为了让 get_latest_observation 能工作，我们需要更新 pos
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.video_num_in_buffer = self.size

    def store_effect(self, action, reward, done):
        """兼容旧接口：为最近存储的帧存储动作、奖励和结束标志。"""
        last_idx = (self.pos - 1) % self.capacity
        self.actions[last_idx] = action
        self.rewards[last_idx] = reward
        self.dones[last_idx] = done

    def store_latest_observation(self, obs_stacked):
        """兼容旧接口：存储最新观测。"""
        pass

    def get_latest_observation(self, k):
        """获取最近 k 帧的堆叠。"""
        if self.size < k:
            # 如果不够，则重复第一帧
            obs = self.observations[0]
            return np.stack([obs] * k, axis=0)
        
        indices = [(self.pos - i - 1) % self.capacity for i in range(k)]
        frames = [self.observations[idx] for idx in reversed(indices)]
        return np.stack(frames, axis=0)

    def sample_per(self, batch_size):
        """优先经验回放采样。"""
        if self.size == 0:
            return None
        
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        obs, act, rew, next_obs, done = self._get_samples(indices)
        return obs, act, rew, next_obs, done, indices, weights

    def sample_n_step_per(self, batch_size, n, gamma):
        """N-步优先采样。"""
        # 简化版：这里假设 n-step 的逻辑在算法端处理，或者在此处预计算
        # 实际实现中通常需要更复杂的索引处理以确保 n-step 期间不跨越 episode
        obs, act, rew, next_obs, done, indices, weights = self.sample_per(batch_size)
        
        # 计算 n-step 奖励和后续状态
        n_rewards = np.zeros(batch_size, dtype=np.float32)
        n_next_obs = []
        n_dones = np.zeros(batch_size, dtype=np.bool_)
        steps_used = np.zeros(batch_size, dtype=np.int32)

        for i, idx in enumerate(indices):
            r = 0
            curr_idx = idx
            actual_n = 0
            for k in range(n):
                r += (gamma ** k) * self.rewards[curr_idx]
                actual_n += 1
                if self.dones[curr_idx]:
                    break
                curr_idx = (curr_idx + 1) % self.capacity
                # 检查是否追上了当前的写入位置
                if curr_idx == self.pos:
                    break
            
            n_rewards[i] = r
            steps_used[i] = actual_n
            # 获取 n 步后的状态
            n_next_idx = (idx + actual_n) % self.capacity
            n_next_obs.append(self._get_stacked_obs(n_next_idx))
            n_dones[i] = self.dones[(idx + actual_n - 1) % self.capacity]

        return obs, act, n_rewards, np.stack(n_next_obs), n_dones, steps_used, indices, weights

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
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = np.abs(error) + 1e-6
        self.max_priority = max(self.max_priority, np.max(np.abs(td_errors)))

    def can_sample(self, batch_size):
        return self.size >= max(batch_size, self.frame_history_len)

    def can_sample_n_step(self, batch_size, n):
        return self.size >= max(batch_size, self.frame_history_len + n)

    def __len__(self):
        return self.size
