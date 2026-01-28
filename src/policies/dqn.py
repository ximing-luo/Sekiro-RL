import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
from src.policies.base.agent import BaseAgent
import configs.config as config

class DQN(BaseAgent):
    """
    DQN 算法实现，遵循框架接口。
    """
    def __init__(self, model_fn, action_dim, buffer, device="cuda", lr=1e-4, gamma=0.99, target_update_freq=1000, n_step_rewards=1):
        super().__init__(action_dim, device)
        self.buffer = buffer
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.n_step_rewards = n_step_rewards
        
        # 网络初始化
        self.eval_net = model_fn().to(self.device)
        self.target_net = model_fn().to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        
        self.optimizer_lock = threading.Lock()
        self.last_loss = 0.0
        self.optimize_count = 0

    def act(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        
        # 统一处理 numpy 数组 (期望输入形状: [k, H, W, C])
        if isinstance(state, np.ndarray):
            state_t = torch.from_numpy(state).float().to(self.device)
        else:
            state_t = torch.FloatTensor(state).to(self.device)
        
        # 转换形状: [k, H, W, C] -> [k, C, H, W] -> [1, k*C, H, W]
        if state_t.ndim == 4:
            k, H, W, C = state_t.shape
            state_t = state_t.permute(0, 3, 1, 2).reshape(1, k*C, H, W)
        else:
            # 备选处理，以防输入已经是 [C_in, H, W]
            state_t = state_t.unsqueeze(0)
            
        state_t = state_t / 255.0
            
        with torch.no_grad():
            q_values = self.eval_net(state_t)
            self._last_q = q_values[0].detach().cpu().numpy()
            return torch.argmax(q_values).item()

    def record(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, done)

    def learn(self, batch_size=32):
        if self.n_step_rewards > 1:
            if not self.buffer.can_sample_n_step(batch_size, self.n_step_rewards):
                return
        elif not self.buffer.can_sample(batch_size):
            return

        # 尝试获取锁，如果已经在优化中则跳过，避免线程积压
        if not self.optimizer_lock.acquire(blocking=False):
            return

        try:
            # 采样
            if self.n_step_rewards > 1:
                obs, act, rew, next_obs, done, steps_used, idxes, weights = self.buffer.sample_n_step_per(batch_size, self.n_step_rewards, self.gamma)
                gamma_power = self.gamma ** steps_used
            else:
                obs, act, rew, next_obs, done, idxes, weights = self.buffer.sample_per(batch_size)
                gamma_power = self.gamma

            obs_t = torch.FloatTensor(obs).to(self.device)
            next_obs_t = torch.FloatTensor(next_obs).to(self.device)
            
            # 形状转换: (B, k, H, W, C) -> (B, k, C, H, W) -> (B, k*C, H, W)
            def process_batch(t):
                B, k, H, W, C = t.shape
                return t.permute(0, 1, 4, 2, 3).reshape(B, k*C, H, W) / 255.0

            obs_t = process_batch(obs_t)
            next_obs_t = process_batch(next_obs_t)
            
            act_t = torch.LongTensor(act).to(self.device)
            rew_t = torch.FloatTensor(rew).to(self.device)
            done_t = torch.FloatTensor(done).to(self.device)
            w_t = torch.FloatTensor(weights).to(self.device)
            gamma_t = torch.FloatTensor(gamma_power).to(self.device) if isinstance(gamma_power, np.ndarray) else gamma_power

            # 计算当前 Q 值
            q_values = self.eval_net(obs_t)
            q_sa = q_values.gather(1, act_t.unsqueeze(1)).squeeze(1)

            # 计算目标 Q 值 (Double DQN)
            with torch.no_grad():
                next_q_eval = self.eval_net(next_obs_t)
                next_act = next_q_eval.argmax(1)
                next_q_tgt = self.target_net(next_obs_t)
                next_q_sa = next_q_tgt.gather(1, next_act.unsqueeze(1)).squeeze(1)
                target = rew_t + gamma_t * (1.0 - done_t) * next_q_sa

            # 计算损失并更新
            td_errors = (q_sa - target).detach().cpu().numpy()
            self.buffer.update_priorities(idxes, td_errors)
            
            loss = (w_t * self.criterion(q_sa, target)).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.last_loss = loss.item()
            self.optimize_count += 1
            
            if self.optimize_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            
            return self.last_loss
        finally:
            self.optimizer_lock.release()

    def save(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load(self, path):
        self.eval_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.eval_net.state_dict())
