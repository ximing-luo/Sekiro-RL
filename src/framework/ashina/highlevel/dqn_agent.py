import os
import torch
import numpy as np
from .base import BaseAgent
from ..algorithm.modelfree import DQNPolicy, DQNAlgorithm
from ..trainer import OffPolicyTrainer
from ..data.collector import Collector
from ..data.batch import Batch
import configs.config as config

class DQNAgent(BaseAgent):
    """
    Ashina 框架下的 DQN 代理。
    作为高层 Facade，协调 Algorithm, Collector 和 Trainer。
    """
    def __init__(self, env, action_dim, buffer, model_file=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(action_dim, device)
        
        # 架构性规范：从 config 对象的层级结构获取配置
        self.model_file = model_file or config.cfg.path.model_path
        
        # 1. 实例化模型 (架构性修复：使用通用的 Sequential 结构，降低对外部特定模型的耦合)
        # env 必须具有 observation_space.shape
        obs_shape = env.observation_space.shape
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(obs_shape), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )
        
        # 2. 实例化策略 (Data Plane)
        self.policy = DQNPolicy(
            model=self.model,
            action_dim=action_dim,
            device=device
        )
        
        # 3. 实例化算法 (Control Plane)
        self.algorithm = DQNAlgorithm(
            policy=self.policy,
            lr=config.cfg.train.learning_rate,
            gamma=config.cfg.train.gamma,
            target_update_freq=1000,
            device=device
        )
        
        # 4. 实例化 Collector
        self.collector = Collector(policy=self.algorithm, env=env, buffer=buffer)
        
        # 5. 实例化训练器
        self.trainer = OffPolicyTrainer(
            algorithm=self.algorithm,
            train_collector=self.collector,
            batch_size=config.cfg.train.batch_size
        )
        self._last_loss = 0.0

    def learn(self):
        loss = self.trainer.train_step()
        if loss is not None:
            self._last_loss = loss
        return loss

    def act(self, state, epsilon=0.0):
        """
        实现 BaseAgent 的 act 接口。
        """
        # 设置探索率
        self.algorithm.policy.set_eps(epsilon)
        
        # 正常推理 (探索逻辑在 Policy 内部处理)
        batch = Batch(obs=np.array([state]))
        result = self.algorithm(batch)
        return result.act.item()

    def record(self, state, action, reward, next_state, done):
        """
        如果需要手动记录数据到 Buffer
        """
        self.collector.buffer.add(state, action, reward, done)

    def save(self, path=None):
        torch.save(self.policy.model.state_dict(), path or self.model_file)

    def load(self, path=None):
        self.policy.model.load_state_dict(torch.load(path or self.model_file, map_location=self.device))
        self.algorithm.sync_target()

    def train(self):
        self.algorithm.policy.train()

    def eval(self):
        self.algorithm.policy.eval()

    @property
    def optimize_count(self):
        return self.algorithm.optimize_count
    
    @property
    def last_loss(self):
        return self._last_loss
