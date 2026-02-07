import os
import torch
import numpy as np
from src.framework.ashina.common.base_agent import BaseAgent
from src.framework.ashina.policies.dqn_policy import DQNPolicy
from src.framework.ashina.trainers.off_policy_trainer import OffPolicyTrainer
from src.models.simple_dqn import ddqn_simple
import configs.config as config

class DQNAgent(BaseAgent):
    """
    Ashina 框架下的 DQN 代理。
    作为高层 Facade，协调 Policy, Trainer 和 Buffer。
    """
    def __init__(self, img_width, img_height, action_dim, buffer, model_file=None, n_step_rewards: int = 1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(action_dim, device)
        
        self.model_file = model_file or config.MODEL_PATH
        self.buffer = buffer
        
        # 1. 实例化模型
        self.model = ddqn_simple(in_channels=config.FRAME_HISTORY_LEN, num_actions=action_dim)
        
        # 2. 实例化策略 (灵魂)
        self.policy = DQNPolicy(
            model=self.model,
            action_dim=action_dim,
            device=device,
            lr=config.LR,
            gamma=config.GAMMA,
            target_update_freq=config.TARGET_UPDATE_FREQ
        )
        
        # 3. 实例化训练器 (执行)
        self.trainer = OffPolicyTrainer(
            policy=self.policy,
            buffer=buffer,
            batch_size=config.BATCH_SIZE,
            n_step=n_step_rewards,
            gamma=config.GAMMA
        )

    def act(self, state, epsilon=0.0):
        # 处理输入状态
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
            
        # 期望形状: [k, H, W, C] -> [k*C, H, W]
        if state.ndim == 4:
            k, H, W, C = state.shape
            state = state.transpose(0, 3, 1, 2).reshape(k*C, H, W)
            
        return self.policy.get_action(state, epsilon)

    def record(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, done)

    def learn(self):
        return self.trainer.train_step()

    def save(self, path=None):
        torch.save(self.policy.eval_net.state_dict(), path or self.model_file)

    def load(self, path=None):
        self.policy.eval_net.load_state_dict(torch.load(path or self.model_file, map_location=self.device))
        self.policy.target_net.load_state_dict(self.policy.eval_net.state_dict())

    def train(self):
        self.policy.eval_net.train()

    def eval(self):
        self.policy.eval_net.eval()

    @property
    def optimize_count(self):
        return self.policy.optimize_count
    
    @property
    def last_loss(self):
        return self.policy.last_loss

    @property
    def last_q(self):
        return getattr(self.policy, '_last_q', None)

