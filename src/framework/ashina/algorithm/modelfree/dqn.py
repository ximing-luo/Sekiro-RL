import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Any, Dict, Optional, Union
from ..base import Policy, Algorithm
from ...data.batch import Batch

class DQNPolicy(Policy):
    """
    DQN 策略实现 (Data Plane)
    职责：接收观测值，输出包含动作和 Q 值的 Batch。
    """
    def __init__(
        self, 
        model: nn.Module, 
        action_dim: int, 
        device: Union[str, torch.device] = "cuda"
    ):
        super().__init__(action_dim, device)
        self.model = model.to(self.device)
        self.eps = 0.0

    def set_eps(self, eps: float) -> None:
        """设置 epsilon 探索率。"""
        self.eps = eps

    def forward(
        self, 
        batch: Batch, 
        state: Optional[Any] = None, 
        **kwargs: Any
    ) -> Batch:
        # 确定性输入处理：直接转换
        obs = torch.as_tensor(batch.obs, device=self.device, dtype=torch.float32)
        
        # 确定性预处理
        if obs.max() > 1.0:
            obs = obs / 255.0
            
        # 处理数据输入（增加 batch 维度）
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
            
        q_values = self.model(obs)
        
        # epsilon-greedy 探索逻辑内聚到策略层
        if self.training and np.random.rand() < self.eps:
            # 探索 (Exploration): 生成随机动作
            act = torch.randint(0, self.action_dim, (q_values.shape[0],), device=self.device)
        else:
            # 选择当前 Q 值最大的动作
            act = q_values.argmax(dim=-1)
        
        return Batch(q_values=q_values, act=act, state=state)

class DQNAlgorithm(Algorithm):
    """
    DQN 算法实现 (Control Plane)
    职责：管理学习过程，包括损失计算、优化和目标网络同步。
    """
    def __init__(
        self,
        policy: DQNPolicy,
        lr: float = 1e-4,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        device: Union[str, torch.device] = "cuda"
    ):
        super().__init__(policy, device)
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        # 目标网络
        self.target_net = copy.deepcopy(self.policy.model).to(self.device)
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.optimize_count = 0

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        """
        DQN 学习步骤。
        """
        batch.to_torch(self.device)
        
        obs = batch.obs
        act = batch.act
        rew = batch.rew
        next_obs = batch.obs_next
        done = batch.done
        
        # 获取可选权重 (用于 PER)
        weights = batch.weight
        if weights is None:
            weights = torch.ones_like(rew)
            
        # 1. 计算当前 Q 值 (维度收敛在 Batch 处理或模型输出层)
        q_values = self.policy.model(obs).flatten(1)
        q_sa = q_values.gather(1, act.view(-1, 1).long()).squeeze(1)

        # 2. 计算目标 Q 值 (Double DQN)
        with torch.no_grad():
            next_q_eval = self.policy.model(next_obs).flatten(1)
            next_act = next_q_eval.argmax(1, keepdim=True)
            
            next_q_tgt = self.target_net(next_obs).flatten(1)
            next_q_sa = next_q_tgt.gather(1, next_act).squeeze(1)
            
            target = rew.flatten() + self.gamma * (1.0 - done.to(torch.float32).flatten()) * next_q_sa.flatten()

        # 3. 计算损失并更新
        td_errors = torch.abs(q_sa - target).detach().cpu().numpy()
        loss = (weights.flatten() * self.criterion(q_sa, target)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.optimize_count += 1
        if self.optimize_count % self.target_update_freq == 0:
            self.sync_target()
            
        return {
            "loss": loss.item(), 
            "td_errors": td_errors,
            "q_avg": q_sa.mean().item()
        }

    def sync_target(self) -> None:
        """硬同步目标网络。"""
        self.target_net.load_state_dict(self.policy.model.state_dict())
