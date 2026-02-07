import torch
import torch.nn as nn
import numpy as np
import threading
from src.framework.ashina.common.base_policy import BasePolicy

class DQNPolicy(BasePolicy):
    """
    DQN 策略逻辑。
    负责计算 Q 值、动作选择和损失计算。
    """
    def __init__(self, model, action_dim, device="cuda", lr=1e-4, gamma=0.99, target_update_freq=1000):
        super().__init__(action_dim, device)
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        # 网络
        self.eval_net = model.to(self.device)
        self.target_net = type(model)(**model.init_args).to(self.device) if hasattr(model, 'init_args') else torch.hub.load('pytorch/vision:v0.10.0', 'resnet18') # 简化处理，实际应从配置克隆
        # 实际更健壮的方法是使用 copy.deepcopy(model)
        import copy
        self.target_net = copy.deepcopy(model).to(self.device)
        
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        
        self.optimizer_lock = threading.Lock()
        self.optimize_count = 0
        self.last_loss = 0.0

    def forward(self, obs):
        """输入 obs [B, C, H, W]，输出 Q 值"""
        return self.eval_net(obs)

    def get_action(self, obs, epsilon=0.0):
        """
        epsilon-greedy 动作选择
        obs 形状: [k*C, H, W] (未加 Batch 维度)
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        
        # 预处理 obs
        if isinstance(obs, np.ndarray):
            obs_t = torch.from_numpy(obs).float().to(self.device)
        else:
            obs_t = obs.to(self.device)
            
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
            
        # 归一化
        if obs_t.max() > 1.0:
            obs_t = obs_t / 255.0
            
        with torch.no_grad():
            q_values = self.forward(obs_t)
            self._last_q = q_values[0].detach().cpu().numpy()
            return torch.argmax(q_values).item()

    def update(self, batch_data):
        """
        batch_data: (obs, act, rew, next_obs, done, weights, gamma_power)
        """
        if not self.optimizer_lock.acquire(blocking=False):
            return None

        try:
            obs, act, rew, next_obs, done, weights, gamma_power = batch_data
            
            # 转换为 Tensor
            obs_t = torch.FloatTensor(obs).to(self.device)
            next_obs_t = torch.FloatTensor(next_obs).to(self.device)
            act_t = torch.LongTensor(act).to(self.device)
            rew_t = torch.FloatTensor(rew).to(self.device)
            done_t = torch.FloatTensor(done).to(self.device)
            w_t = torch.FloatTensor(weights).to(self.device)
            gamma_t = torch.FloatTensor(gamma_power).to(self.device) if isinstance(gamma_power, np.ndarray) else gamma_power

            # 计算当前 Q 值
            q_values = self.forward(obs_t)
            q_sa = q_values.gather(1, act_t.unsqueeze(1)).squeeze(1)

            # 计算目标 Q 值 (Double DQN)
            with torch.no_grad():
                next_q_eval = self.eval_net(next_obs_t)
                next_act = next_q_eval.argmax(1)
                next_q_tgt = self.target_net(next_obs_t)
                next_q_sa = next_q_tgt.gather(1, next_act.unsqueeze(1)).squeeze(1)
                target = rew_t + gamma_t * (1.0 - done_t) * next_q_sa

            # 计算损失
            td_errors = (q_sa - target).detach().cpu().numpy()
            loss = (w_t * self.criterion(q_sa, target)).mean()
            
            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.optimize_count += 1
            if self.optimize_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            
            self.last_loss = loss.item()
            return self.last_loss, td_errors
        finally:
            self.optimizer_lock.release()
