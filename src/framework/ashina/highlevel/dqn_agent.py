import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass
from .base import BaseAgent
from ..algorithm.modelfree import DQNAlgorithm
from ..trainer import OffPolicyTrainer
from ..data import Batch, Collector, ReplayBuffer
from ..env import VectorEnv


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    target_update_freq: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 2048
    steps_per_iter: int = 24
    epochs: int = 4


class DQNAgent(BaseAgent):
    """
    DQN 代理：纯并行版本。
    支持多环境并行采集和学习。
    """
    def __init__(
        self,
        env: VectorEnv,
        action_dim: int,
        buffer: ReplayBuffer,
        params: DQNConfig = None,
        model: Optional[torch.nn.Module] = None,
        model_file: Optional[str] = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(action_dim, device)

        self.model_file = model_file
        p = params or DQNConfig()

        obs_batch, _ = env.reset()
        input_dim = int(np.prod(obs_batch[0].shape))

        self.algorithm = DQNAlgorithm(
            action_dim=action_dim,
            input_dim=input_dim,
            model=model,
            device=device,
            lr=p.lr,
            gamma=p.gamma,
            target_update_freq=p.target_update_freq,
            epsilon_start=p.epsilon_start,
            epsilon_end=p.epsilon_end,
            epsilon_decay=p.epsilon_decay,
        )

        self.env = env

        self.collector = Collector(
            algorithm=self.algorithm,
            env=self.env,
            buffer=buffer
        )

        self.trainer = OffPolicyTrainer(
            algorithm=self.algorithm,
            train_collector=self.collector,
            batch_size=p.batch_size,
            steps_per_iter=p.steps_per_iter,
            epochs=p.epochs,
        )

    def act(self, state):
        """单个 state 推理（用于评估）"""
        return self.algorithm(Batch(obs=np.array([state]))).act.item()

    def learn(self):
        return self.trainer.train_iteration()

    def train(self):
        """切换到训练模式"""
        self.algorithm.policy.train()

    def eval(self):
        """切换到评估模式"""
        self.algorithm.policy.eval()

    def save(self, path=None):
        """保存模型"""
        torch.save(self.algorithm.policy.model.state_dict(), path or self.model_file)

    def load(self, path=None):
        """加载模型"""
        self.algorithm.policy.model.load_state_dict(
            torch.load(path or self.model_file, map_location=self.device, weights_only=True)
        )
        self.algorithm.sync_target()

    @property
    def optimize_count(self):
        """优化步数"""
        return self.algorithm.optimize_count

    def close(self):
        """关闭环境"""
        self.env.close()
