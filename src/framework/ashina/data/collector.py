from typing import Any, Dict, Optional
import numpy as np
from .batch import Batch
from .buffer import ReplayBuffer
from ..algorithm.base import Algorithm


class Collector:
    """
    向量化数据采集器：同时采集 N 个环境的数据。
    terminal obs 替换由 VectorEnv 子类的 step() 透明处理。
    """
    def __init__(
        self,
        algorithm: Algorithm,
        env: Any,
        buffer: Optional[ReplayBuffer] = None,
    ):
        self.algorithm = algorithm
        self.env = env
        self.buffer = buffer
        self._obs_batch, _ = env.reset()
        self.env_num = len(self._obs_batch) if hasattr(self._obs_batch, '__len__') else 1

    def reset(self):
        self._obs_batch, _ = self.env.reset()

    def collect_random_steps(self, n_step: int) -> None:
        """随机动作采集"""
        steps_collected = 0
        while steps_collected < n_step:
            actions = self.env.action_space.sample()
            obs_batch, rew_batch, done_batch, trunc_batch, _ = self.env.step(actions)

            for i in range(self.env_num):
                if self.buffer is not None:
                    self.buffer.add(self._obs_batch[i], actions[i], rew_batch[i], obs_batch[i], done_batch[i] or trunc_batch[i])

            self._obs_batch = obs_batch
            steps_collected += self.env_num

    def collect_steps(self, n_step: int) -> Dict[str, Any]:
        """策略采集"""
        all_rewards = []
        steps_collected = 0

        while steps_collected < n_step:
            obs_batch = Batch(obs=self._obs_batch)
            actions = self.algorithm(obs_batch).act
            obs_batch_next, rew_batch, done_batch, trunc_batch, _ = self.env.step(actions)

            for i in range(self.env_num):
                if self.buffer is not None:
                    self.buffer.add(self._obs_batch[i], actions[i], rew_batch[i], obs_batch_next[i], done_batch[i])

                all_rewards.append(rew_batch[i])

            self._obs_batch = obs_batch_next
            steps_collected += self.env_num

        return {"n_step": steps_collected, "reward": np.mean(all_rewards)}

    def collect_episodes(self, n_episode: int) -> Dict[str, Any]:
        """完整 episode 采集"""
        episode_count = 0
        all_rewards = []
        episode_rew = np.zeros(self.env_num)

        while episode_count < n_episode:
            obs_batch = Batch(obs=self._obs_batch)
            actions = self.algorithm(obs_batch).act
            obs_batch_next, rew_batch, done_batch, trunc_batch, _ = self.env.step(actions)

            for i in range(self.env_num):
                if self.buffer is not None:
                    self.buffer.add(self._obs_batch[i], actions[i], rew_batch[i], obs_batch_next[i], done_batch[i])

                episode_rew[i] += rew_batch[i]

                if done_batch[i] or trunc_batch[i]:
                    all_rewards.append(episode_rew[i])
                    episode_rew[i] = 0.0
                    episode_count += 1

            self._obs_batch = obs_batch_next

        return {"n_episode": n_episode, "reward": np.mean(all_rewards)}
