import numpy as np
import torch
from .venvs import VectorEnv


class GymnasiumWrapper(VectorEnv):
    """
    Gymnasium VectorEnv 包装器：处理 torch → numpy 转换
    内联 terminal obs 到 obs_next
    """
    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        obs_next, rew, done, trunc, info = self.env.step(actions)
        # 把 Gymnasium VecEnv 的 terminal obs 内联替换进 obs_next
        final_obs = info.get("final_observation")
        mask = info.get("_final_observation")
        if final_obs is not None and mask is not None:
            obs_next = obs_next.copy()
            obs_next[mask] = np.stack([final_obs[i] for i in np.where(mask)[0]])
        return obs_next, rew, done, trunc, info

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def num_envs(self):
        return self.env.num_envs
