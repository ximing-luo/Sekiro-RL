import os
import torch
import numpy as np
from .base import BaseAgent
from ..policy import DQNPolicy
from ..trainer import OffPolicyTrainer
from ..data.collector import Collector
from src.models.simple_dqn import ddqn_simple
import configs.config as config

class DQNAgent(BaseAgent):
    """
    Ashina 框架下的 DQN 代理。
    作为高层 Facade，协调 Policy, Collector 和 Trainer。
    """
    def __init__(self, env, action_dim, buffer, model_file=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(action_dim, device)
        
        self.model_file = model_file or config.MODEL_PATH
        
        # 1. 实例化模型
        self.model = ddqn_simple(in_channels=config.FRAME_HISTORY_LEN, num_actions=action_dim)
        
        # 2. 实例化策略
        self.policy = DQNPolicy(
            model=self.model,
            action_dim=action_dim,
            device=device,
            lr=config.LR,
            gamma=config.GAMMA,
            target_update_freq=config.TARGET_UPDATE_FREQ
        )
        
        # 3. 实例化 Collector (天授式核心)
        self.collector = Collector(policy=self.policy, env=env, buffer=buffer)
        
        # 4. 实例化训练器
        self.trainer = OffPolicyTrainer(
            policy=self.policy,
            train_collector=self.collector,
            batch_size=config.BATCH_SIZE
        )

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

