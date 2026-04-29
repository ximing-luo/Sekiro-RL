import numpy as np
import torch
from .venvs import VectorEnv


class GymnasiumWrapper(VectorEnv):
    """Gymnasium VectorEnv 包装器：处理 torch tensor → numpy 转换"""
    def __init__(self, vec_env):
        self.vec_env = vec_env
        self._obs_batch = None

    def reset(self, **kwargs):
        obs_batch, info = self.vec_env.reset(**kwargs)
        self._obs_batch = obs_batch
        return obs_batch, info

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        obs_batch, rew_batch, done_batch, trunc_batch, info = self.vec_env.step(actions)
        self._obs_batch = obs_batch
        return obs_batch, rew_batch, done_batch, trunc_batch, info

    def close(self):
        self.vec_env.close()

    @property
    def action_space(self):
        return self.vec_env.action_space

    @property
    def observation_space(self):
        return self.vec_env.observation_space

    @property
    def num_envs(self):
        return self.vec_env.num_envs
