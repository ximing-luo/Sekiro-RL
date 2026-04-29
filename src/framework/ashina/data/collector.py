from typing import Any, Dict, Optional
import numpy as np
from .batch import Batch
from ..algorithm.base import Algorithm
from .buffer import ReplayBuffer


class Collector:
    """
    向量化数据采集器：同时采集 N 个环境的数据。
    采用 Gymnasium 标准 info 格式（dict），不做兼容适配。
    """
    def __init__(
        self,
        algorithm: Algorithm,
        vec_env: Any,
        buffer: Optional[ReplayBuffer] = None,
    ):
        self.algorithm = algorithm
        self.vec_env = vec_env
        self.buffer = buffer
        self._obs_batch, _ = vec_env.reset()
        self.env_num = len(self._obs_batch) if hasattr(self._obs_batch, '__len__') else 1

    def reset(self):
        self._obs_batch, _ = self.vec_env.reset()

    def collect_random_steps(self, n_step: int) -> None:
        """随机动作采集"""
        steps_collected = 0
        while steps_collected < n_step:
            actions = self.vec_env.action_space.sample()
            obs_batch, rew_batch, done_batch, trunc_batch, info = self.vec_env.step(actions)

            for i in range(self.env_num):
                if self.buffer is not None:
                    obs_next = self._get_terminal_obs(info, i, obs_batch[i], done_batch[i], self.env_num)
                    self.buffer.add(self._obs_batch[i], actions[i], rew_batch[i], obs_next, done_batch[i] or trunc_batch[i])

            self._obs_batch = obs_batch
            steps_collected += self.env_num

    def collect_steps(self, n_step: int) -> Dict[str, Any]:
        """策略采集"""
        all_rewards = []
        steps_collected = 0

        while steps_collected < n_step:
            obs_batch = Batch(obs=self._obs_batch)
            actions = self.algorithm(obs_batch).act
            obs_batch_next, rew_batch, done_batch, trunc_batch, info = self.vec_env.step(actions)

            for i in range(self.env_num):
                if self.buffer is not None:
                    obs_next = self._get_terminal_obs(info, i, obs_batch_next[i], done_batch[i], self.env_num)
                    self.buffer.add(self._obs_batch[i], actions[i], rew_batch[i], obs_next, done_batch[i])

                all_rewards.append(rew_batch[i])

            self._obs_batch = obs_batch_next
            steps_collected += self.env_num

        return {"n_step": steps_collected, "rew": np.mean(all_rewards)}

    def collect_episodes(self, n_episode: int) -> Dict[str, Any]:
        """完整 episode 采集"""
        episode_count = 0
        all_rewards = []
        episode_rew = np.zeros(self.env_num)

        while episode_count < n_episode:
            obs_batch = Batch(obs=self._obs_batch)
            actions = self.algorithm(obs_batch).act
            obs_batch_next, rew_batch, done_batch, trunc_batch, info = self.vec_env.step(actions)

            for i in range(self.env_num):
                if self.buffer is not None:
                    obs_next = self._get_terminal_obs(info, i, obs_batch_next[i], done_batch[i], self.env_num)
                    self.buffer.add(self._obs_batch[i], actions[i], rew_batch[i], obs_next, done_batch[i])

                episode_rew[i] += rew_batch[i]

                if done_batch[i] or trunc_batch[i]:
                    all_rewards.append(episode_rew[i])
                    episode_rew[i] = 0.0
                    episode_count += 1

            self._obs_batch = obs_batch_next

        return {"n_episode": n_episode, "rew": np.mean(all_rewards)}

    @staticmethod
    def _get_terminal_obs(info: dict, i: int, default_obs: np.ndarray, done: bool, env_num: int) -> np.ndarray:
        """从 Gymnasium 标准 info 格式提取终止帧"""
        if not done:
            return default_obs
        mask = info.get("_final_observation", np.array([False] * env_num))
        if mask[i]:
            return info["final_observation"][i]
        return default_obs
