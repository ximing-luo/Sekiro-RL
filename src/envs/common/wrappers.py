import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class SekiroSkipFrameStack(gym.Wrapper):
    """
    自定义跳帧堆叠包装器。
    逻辑：每隔 skip 帧取 1 帧，总共堆叠 n_stack 帧。
    
    例如：n_stack=4, skip=3
    缓冲区会保存最近的 12 帧 (3 * 4 = 12)。
    输出的观测值将由索引为 [0, 3, 6, 9] 的帧拼接而成。
    这能覆盖更长的时间跨度，帮助模型识别动作趋势。
    """
    def __init__(self, env, n_stack=4, skip=3):
        super().__init__(env)
        self.n_stack = n_stack
        self.skip = skip
        self.max_len = n_stack * skip
        
        # 内部缓冲区，保存原始图像
        self.frames = deque(maxlen=self.max_len)
        
        # 更新观测空间维度
        low = np.repeat(self.env.observation_space.low, n_stack, axis=0)
        high = np.repeat(self.env.observation_space.high, n_stack, axis=0)
        self.observation_space = spaces.Box(
            low=low, 
            high=high, 
            dtype=self.env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 初始化缓冲区：用第一帧填满
        for _ in range(self.max_len):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # 按照跳帧逻辑采样
        # 索引 [0, 3, 6, 9] 对应从旧到新的跳帧
        sampled_frames = [self.frames[i * self.skip] for i in range(self.n_stack)]
        # 在通道维度 (axis=0) 进行拼接
        return np.concatenate(sampled_frames, axis=0)
