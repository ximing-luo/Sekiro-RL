from __future__ import annotations
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Sequence

from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    """
    基于管理器的强化学习环境类。
    符合 Gymnasium 接口标准：step(), reset()。
    
    对标 Isaac Lab 的 ManagerBasedRLEnv。
    """
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None):
        # 1. 调用父类初始化管理器和仿真 (固化契约：仅传递就绪的 cfg)
        super().__init__(cfg, render_mode)
        self._setup_managers()
        self._configure_spaces()

    def _configure_spaces(self):
        """根据管理器配置物理化推导动作和观测空间。
        
        遵循“逻辑-效能同构”：将探测逻辑从热路径剥离，坍缩为直线属性引用。
        """
        # A. 动作空间：物理指代
        self.action_space = self.action_manager.action_space
        # B. 观测空间：直接使用管理器提供的 Dict 空间
        self.observation_space = self.observation_manager.observation_space

    @property
    def action_dim(self):
        """兼容性属性：返回动作空间总维度。"""
        return sum(self.action_manager.action_term_dim)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Dict, Dict]:
        """Gymnasium 标准重置。"""
        # 遵循 Gymnasium 标准处理 seed (虽然目前 sim 不支持 seed)
        super().reset() # 调用 ManagerBasedEnv.reset
        
        # 获取初始观测
        obs = self.observation_manager.step()
        
        return obs, {}

    def step(self, action: torch.Tensor) -> tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Gymnasium 标准步进。
        
        对齐 Isaac Lab：始终返回 Tensor 以支持多环境向量化。
        """
        return super().step(action)
