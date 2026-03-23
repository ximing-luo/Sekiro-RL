from typing import Dict, Any, Optional
from ..data.collector import Collector
from ..policy.base import BasePolicy

class OffPolicyTrainer:
    """
    天授式 Trainer：调度 Collector 和 Policy。
    """
    def __init__(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Optional[Collector] = None,
        batch_size: int = 64,
        update_per_step: float = 1.0,
    ):
        self.policy = policy
        self.train_collector = train_collector
        self.test_collector = test_collector
        self.batch_size = batch_size
        self.update_per_step = update_per_step

    def train_step(self):
        """执行一轮训练：采样 -> 学习"""
        # 1. 采样数据
        self.train_collector.collect(n_step=1)
        
        # 2. 从 Buffer 采样并学习
        if len(self.train_collector.buffer) >= self.batch_size:
            batch = self.train_collector.buffer.sample(self.batch_size)
            result = self.policy.learn(batch)
            return result["loss"]
        return None
