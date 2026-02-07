import torch
import torch.nn as nn
import numpy as np
import threading
from .base import BasePolicy
from ..data.batch import Batch
from typing import Dict, Any, Optional

class DQNPolicy(BasePolicy):
    """
    DQN 策略逻辑。
    """
    def __init__(self, model, action_dim, device="cuda", lr=1e-4, gamma=0.99, target_update_freq=1000):
        super().__init__(action_dim, device)
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        # 网络
        self.eval_net = model.to(self.device)
        import copy
        self.target_net = copy.deepcopy(model).to(self.device)
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.optimize_count = 0

    def forward(self, batch: Batch, state: Optional[Any] = None) -> Batch:
        """
        天授风格 forward：输入 Batch，输出包含 logits 和 act 的 Batch
        """
        obs = batch.obs
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        
        # 归一化处理
        if obs.max() > 1.0:
            obs = obs / 255.0
            
        logits = self.eval_net(obs)
        act = logits.argmax(dim=1)
        
        return Batch(logits=logits, act=act, state=state)

    def learn(self, batch: Batch) -> Dict[str, float]:
        """
        天授风格 learn：从采样 Batch 中学习
        """
        # 转换 Batch 数据为 Tensor
        batch.to_torch(self.device)
        
        obs = batch.obs
        act = batch.act
        rew = batch.rew
        next_obs = batch.obs_next
        done = batch.done
        weights = getattr(batch, "weight", torch.ones_like(rew))
        gamma_power = getattr(batch, "gamma_p", self.gamma)

        # 计算当前 Q 值
        q_values = self.eval_net(obs)
        q_sa = q_values.gather(1, act.unsqueeze(1).long()).squeeze(1)

        # 计算目标 Q 值 (Double DQN)
        with torch.no_grad():
            next_q_eval = self.eval_net(next_obs)
            next_act = next_q_eval.argmax(1)
            next_q_tgt = self.target_net(next_obs)
            next_q_sa = next_q_tgt.gather(1, next_act.unsqueeze(1)).squeeze(1)
            target = rew + gamma_power * (1.0 - done) * next_q_sa

        # 计算损失
        td_errors = torch.abs(q_sa - target).detach().cpu().numpy()
        loss = (weights * self.criterion(q_sa, target)).mean()
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.optimize_count += 1
        if self.optimize_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            
        return {"loss": loss.item(), "td_errors": td_errors}

