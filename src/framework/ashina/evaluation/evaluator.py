from typing import Any, Dict, List, Optional, Union
import numpy as np
from ..algorithm.base import Algorithm
from ..data.collector import Collector

class Evaluator:
    """
    评估器。
    评估通常通过独立的 Collector 在测试环境中运行若干 episode。
    """
    def __init__(
        self,
        algorithm: Algorithm,
        test_collector: Collector,
    ):
        self.algorithm = algorithm
        self.test_collector = test_collector

    def evaluate(self, n_episode: int = 10) -> Dict[str, Any]:
        """
        运行指定数量的 episode 并返回平均奖励。
        """
        # 切换到评估模式 (关闭探索，如 EpsilonGreedy 或 Dropout)
        self.algorithm.policy.eval()

        # 使用 Collector 的 n_episode 功能
        result = self.test_collector.collect(n_episode=n_episode)
        
        # 恢复训练模式
        self.algorithm.policy.train()
        
        return {
            "test_reward": result["rew"],
            "n_episode": n_episode
        }
