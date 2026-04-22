from abc import ABC, abstractmethod
from typing import Any, List, Optional, Callable
import numpy as np

class BaseVectorEnv(ABC):
    """
    矢量化环境基类。对标 Tianshou 2.0 的 venvs.py。
    """
    def __init__(self, env_fns: List[Callable[[], Any]]):
        self.env_num = len(env_fns)
        self.envs = [fn() for fn in env_fns]

    @abstractmethod
    def reset(self, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def step(self, actions: np.ndarray) -> Any:
        pass

    def close(self) -> None:
        for env in self.envs:
            env.close()

class DummyVectorEnv(BaseVectorEnv):
    """
    串行执行的矢量化环境包装。
    """
    def reset(self, **kwargs: Any) -> Any:
        results = [env.reset(**kwargs) for env in self.envs]
        # 假设返回格式为 (obs, info)
        obs_list = np.array([r[0] for r in results])
        info_list = [r[1] for r in results]
        return obs_list, info_list

    def step(self, actions: np.ndarray) -> Any:
        obs_list, rew_list, done_list, trunc_list, info_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, done, trunc, info = env.step(actions[i])
            if done or trunc:
                obs, info = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            trunc_list.append(trunc)
            info_list.append(info)
        return (
            np.array(obs_list),
            np.array(rew_list),
            np.array(done_list),
            np.array(trunc_list),
            info_list
        )
