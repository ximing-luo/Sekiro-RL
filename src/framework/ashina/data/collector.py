from typing import Any, Dict, Optional
import numpy as np
from .batch import Batch
from ..policy.base import BasePolicy
from .replay_buffer import ReplayBuffer

class Collector:
    """
    天授式 Collector：连接策略与环境的桥梁。
    职责：
    1. 执行策略在环境中运行。
    2. 收集数据并存入 ReplayBuffer。
    """
    def __init__(
        self,
        policy: BasePolicy,
        env: Any,
        buffer: Optional[ReplayBuffer] = None,
    ):
        self.policy = policy
        self.env = env
        self.buffer = buffer
        self._obs = None
        self.reset()

    def reset(self):
        """重置环境并初始化第一个观测值"""
        self._obs, _ = self.env.reset()

    def collect(self, n_step: int = 1) -> Dict[str, Any]:
        """
        执行 n 步采样并存入 buffer。
        """
        rewards = []
        for _ in range(n_step):
            # 1. 策略前向传播得到动作
            batch = Batch(obs=np.array([self._obs]))
            result = self.policy.forward(batch)
            action = result.act.item()

            # 2. 与环境交互
            obs_next, rew, done, truncated, info = self.env.step(action)
            
            # 3. 存入 Buffer
            if self.buffer is not None:
                self.buffer.add(self._obs, action, rew, done)

            rewards.append(rew)
            self._obs = obs_next

            if done or truncated:
                self.reset()

        return {
            "n_step": n_step,
            "rew": np.mean(rewards) if rewards else 0.0,
        }
