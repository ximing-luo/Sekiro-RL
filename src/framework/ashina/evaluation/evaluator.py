from typing import Any, Dict
from ..algorithm.base import Algorithm

class Evaluator:
    """
    评估器。
    """
    def __init__(self, algorithm: Algorithm, test_collector: Any):
        self.algorithm = algorithm
        self.test_collector = test_collector

    def evaluate(self, n_episode: int = 10) -> Dict[str, Any]:
        self.algorithm.policy.eval()
        result = self.test_collector.collect_episodes(n_episode=n_episode)
        self.algorithm.policy.train()
        return {"test_reward": result["rew"], "n_episode": n_episode}
