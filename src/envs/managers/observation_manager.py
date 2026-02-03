from typing import Dict, Any, Optional
import numpy as np
from collections import deque
from src.envs.base.manager_based_env_cfg import ObservationTermCfg

class ObservationManager:
    """
    观测管理器：实现基于术语的观测提取，并提供帧堆叠支持。
    """
    def __init__(self, cfg: Dict[str, ObservationTermCfg]):
        self.cfg = cfg
        # 帧缓冲区，用于动态堆叠。最大长度为 n_stack * skip (默认 4*3=12)
        self.frame_buffer = deque(maxlen=12)

    def compute_observations(self, env):
        """遍历配置中的所有观测项。"""
        observations = {}
        for name, term_cfg in self.cfg.items():
            val = term_cfg.func(env=env, **term_cfg.params)
            observations[name] = val
            
            # 如果是主策略图像，存入缓冲区
            if name == "policy" and val is not None:
                # 统一转换为 CHW 格式存储
                if val.ndim == 3 and val.shape[-1] == 3:
                    val = val.transpose(2, 0, 1)
                self.frame_buffer.append(val)
                
        return observations

    def get_latest_stacked_frames(self, n_stack: int = 4, skip: int = 3) -> np.ndarray:
        """
        从缓冲区获取跳帧堆叠后的图像。
        逻辑：每隔 skip 帧取 1 帧，总共堆叠 n_stack 帧。
        """
        # 如果缓冲区为空，返回一个全零的保底张量，防止下游崩溃
        if len(self.frame_buffer) == 0:
            # 假设标准分辨率为 270x480
            return np.zeros((n_stack * 3, 270, 480), dtype=np.uint8)
            
        # 如果缓冲区还没填满，用第一帧补齐
        current_frames = list(self.frame_buffer)
        while len(current_frames) < n_stack * skip:
            current_frames.insert(0, current_frames[0])
            
        # 按照跳帧逻辑采样
        # 索引 [0, 3, 6, 9] 对应从旧到新的跳帧 (假设 skip=3)
        sampled_frames = [current_frames[i * skip] for i in range(n_stack)]
        
        # 在通道维度 (axis=0) 进行拼接
        return np.concatenate(sampled_frames, axis=0)
