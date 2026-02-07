import time
import numpy as np

class OffPolicyTrainer:
    """
    离线策略算法训练器。
    负责协调 Buffer 采样和 Policy 更新。
    """
    def __init__(self, policy, buffer, batch_size=32, n_step=1, gamma=0.99):
        self.policy = policy
        self.buffer = buffer
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma

    def train_step(self):
        """执行一个训练步"""
        if self.n_step > 1:
            if not self.buffer.can_sample_n_step(self.batch_size, self.n_step):
                return None
            obs, act, rew, next_obs, done, steps_used, idxes, weights = self.buffer.sample_n_step_per(
                self.batch_size, self.n_step, self.gamma)
            gamma_power = self.gamma ** steps_used
        else:
            if not self.buffer.can_sample(self.batch_size):
                return None
            obs, act, rew, next_obs, done, idxes, weights = self.buffer.sample_per(self.batch_size)
            gamma_power = self.gamma

        # 数据预处理 (如果是图像数据，进行形状变换)
        obs = self._process_obs(obs)
        next_obs = self._process_obs(next_obs)

        # 更新策略
        batch_data = (obs, act, rew, next_obs, done, weights, gamma_power)
        result = self.policy.update(batch_data)
        
        if result is not None:
            loss, td_errors = result
            # 更新 Buffer 优先级
            self.buffer.update_priorities(idxes, td_errors)
            return loss
        return None

    def _process_obs(self, obs):
        """处理图像数据: (B, k, H, W, C) -> (B, k*C, H, W)"""
        if obs.ndim == 5:
            B, k, H, W, C = obs.shape
            # 这里的转换逻辑应与具体项目对齐
            # (B, k, H, W, C) -> (B, k, C, H, W) -> (B, k*C, H, W)
            obs = obs.transpose(0, 1, 4, 2, 3).reshape(B, k*C, H, W)
            return obs / 255.0
        return obs
