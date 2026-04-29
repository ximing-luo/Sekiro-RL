import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Union
from ..base import Algorithm, Policy
from ...data.batch import Batch

class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) 核心模型。
    包含特征提取网络、前向模型和逆向模型。
    """
    def __init__(self, feature_net: nn.Module, feature_dim: int, action_dim: int, hidden_sizes=[512]):
        super().__init__()
        self.feature_net = feature_net
        self.feature_dim = feature_dim
        
        # 逆向模型 (Inverse Model): 从 (phi(s), phi(s')) 预测动作
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], action_dim)
        )
        
        # 前向模型 (Forward Model): 从 (phi(s), action) 预测 phi(s')
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], feature_dim)
        )

    def forward(self, obs, next_obs, action):
        # 处理 action 维度 (One-hot 编码)
        if action.ndim == 1 or (action.ndim == 2 and action.shape[1] == 1):
            action = F.one_hot(action.long().flatten(), num_classes=self.inverse_net[-1].out_features).float()
            
        phi_s = self.feature_net(obs).flatten(1)
        phi_s_next = self.feature_net(next_obs).flatten(1)
        
        # 逆向预测动作
        pred_action = self.inverse_net(torch.cat([phi_s, phi_s_next], dim=1))
        
        # 前向预测下一状态特征
        pred_phi_s_next = self.forward_net(torch.cat([phi_s, action], dim=1))
        
        return phi_s_next, pred_phi_s_next, pred_action

class ICMAlgorithmWrapper(Algorithm):
    """
    ICM 算法包装器 (Control Plane)。
    包装一个现有的离策算法，注入内在奖励逻辑。
    """
    def __init__(
        self,
        wrapped_algorithm: Algorithm,
        model: ICMModule,
        lr: float = 1e-4,
        lr_scale: float = 1.0,
        reward_scale: float = 0.01,
        forward_loss_weight: float = 0.2,
    ):
        super().__init__(wrapped_algorithm.policy, wrapped_algorithm.device)
        self.wrapped_algorithm = wrapped_algorithm
        self.icm_model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.icm_model.parameters(), lr=lr * lr_scale)
        
        self.reward_scale = reward_scale
        self.forward_loss_weight = forward_loss_weight

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        """
        学习逻辑：计算内在奖励 -> 更新基础算法 -> 更新 ICM 模型。
        """
        batch.to_torch(self.device)
        obs, next_obs, act = batch.obs, batch.obs_next, batch.act
        
        # 1. 计算 ICM 前向预测误差作为内在奖励
        phi_s_next, pred_phi_s_next, pred_action = self.icm_model(obs, next_obs, act)
        
        # 前向损失: MSE (用于计算内在奖励)
        forward_loss_batch = F.mse_loss(pred_phi_s_next, phi_s_next, reduction='none').mean(1)
        intrinsic_reward = self.reward_scale * forward_loss_batch.detach()
        
        # 将内在奖励注入 Batch 用于基础算法学习
        batch.rew = batch.rew + intrinsic_reward.unsqueeze(1)
        
        # 2. 调用被包装算法的学习逻辑 (如 DQN)
        result = self.wrapped_algorithm.learn(batch, **kwargs)
        
        # 3. 计算并更新 ICM 自身的损失
        # 逆向损失: 交叉熵 (预测动作)
        inverse_loss = F.cross_entropy(pred_action, act.long().flatten())
        # 前向损失均值
        forward_loss = forward_loss_batch.mean()
        
        icm_loss = (1 - self.forward_loss_weight) * inverse_loss + self.forward_loss_weight * forward_loss
        
        self.optimizer.zero_grad()
        icm_loss.backward()
        self.optimizer.step()
        
        # 合并统计指标
        result.update({
            "icm_loss": icm_loss.item(),
            "icm_inv_loss": inverse_loss.item(),
            "icm_fwd_loss": forward_loss.item(),
            "intrinsic_reward": intrinsic_reward.mean().item()
        })
        return result

    def sync_target(self):
        self.wrapped_algorithm.sync_target()

    def state_dict(self):
        return {
            "wrapped": self.wrapped_algorithm.state_dict(),
            "icm": self.icm_model.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.wrapped_algorithm.load_state_dict(state_dict["wrapped"])
        self.icm_model.load_state_dict(state_dict["icm"])
