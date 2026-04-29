from typing import Dict, Any, Optional
from ..algorithm.base import Algorithm
from ..data.collector import Collector

class OffPolicyTrainer:
    """
    Trainer：调度 Collector 和 Algorithm。
    迭代式训练：先大规模采集，再反复学习。
    """
    def __init__(
        self,
        algorithm: Algorithm,
        train_collector: Collector,
        test_collector: Optional[Collector] = None,
        batch_size: int = 2048,
        steps_per_iter: int = 24,
        epochs: int = 4,
    ):
        self.algorithm = algorithm
        self.train_collector = train_collector
        self.test_collector = test_collector
        self.buffer = self.train_collector.buffer

        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_iter = steps_per_iter
        self.env_num = self.train_collector.env_num
        self.n_batches = (self.env_num * self.steps_per_iter) // self.batch_size

        self._step = 0

    def train_iteration(self) -> Dict[str, Any]:
        """一次迭代：采集 steps_per_iter 步 → 反复学习 epochs 轮"""
        self.algorithm.pre_collect(self._step)
        self._step += 1

        collect_result = self.train_collector.collect_steps(n_step=self.steps_per_iter)

        for _ in range(self.epochs):
            for _ in range(self.n_batches):
                batch = self.buffer.sample(self.batch_size)
                learn_result = self.algorithm.learn(batch)

        return {
            "reward": collect_result["reward"],
            "loss": learn_result["loss"],
            "td_errors": learn_result["td_errors"],
            "q_avg": learn_result["q_avg"],
        }

    def eval_step(self, n_episode: int = 5) -> Dict[str, Any]:
        return self.test_collector.collect_episodes(n_episode=n_episode)
