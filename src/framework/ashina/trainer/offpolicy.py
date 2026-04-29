from typing import Dict, Any, Optional
from ..algorithm.base import Algorithm

class OffPolicyTrainer:
    """
    Trainer：调度 Collector 和 Algorithm。
    迭代式训练：先大规模采集，再反复学习。
    """
    def __init__(
        self,
        algorithm: Algorithm,
        train_collector: Any,
        test_collector: Optional[Any] = None,
        batch_size: int = 2048,
    ):
        self.algorithm = algorithm
        self.train_collector = train_collector
        self.test_collector = test_collector
        self.batch_size = batch_size
        self._step = 0

    def train_iteration(self, steps_per_iter: int, learn_epochs: int = 4) -> Dict[str, Any]:
        """
        一次迭代：采集 steps_per_iter 步 → 反复学习 learn_epochs 轮。
        steps_per_iter: 每个环境跑多少步
        learn_epochs:   采集的数据反复学几轮
        """
        self.algorithm.pre_collect(self._step)
        self._step += 1

        collect_result = self.train_collector.collect_steps(n_step=steps_per_iter)
        total_samples = self.train_collector.env_num * steps_per_iter
        batches_per_epoch = total_samples // self.batch_size

        if batches_per_epoch == 0:
            return {}

        buffer = self.train_collector.buffer
        for epoch in range(learn_epochs):
            for _ in range(batches_per_epoch):
                batch = buffer.sample(self.batch_size)
                self.algorithm.learn(batch)

        return {
            "rew": collect_result.get("rew", 0.0),
            "batches_per_epoch": batches_per_epoch,
            "epochs": learn_epochs,
        }

    def eval_step(self, n_episode: int = 5) -> Dict[str, Any]:
        return self.test_collector.collect_episodes(n_episode=n_episode)
