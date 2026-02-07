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
        # 1. 调用父类初始化管理器和仿真
        super().__init__(cfg, render_mode)
        self._setup_managers()
        
        # 2. 定义 Gymnasium 空间 (基于管理器提供的信息)
        self._configure_spaces()

    def _configure_spaces(self):
        """根据管理器配置自动推导动作和观测空间。"""
        # A. 动作空间
        # 目前假设是 MultiDiscrete (只狼常用)
        # TODO: 根据 ActionManager 的具体类型动态判断
        self.action_space = spaces.MultiDiscrete(self.action_manager.action_term_dim)
        
        # B. 观测空间 (目前支持 Dict 模式以适配图像+遥测)
        obs_dict = {}
        # 图像空间
        obs_dict["image"] = spaces.Box(
            low=0, high=1.0, 
            shape=(3, self.cfg.scene.observation_h, self.cfg.scene.observation_w), 
            dtype=np.float32
        )
        # 遥测空间 (10个归一化指标)
        obs_dict["telemetry"] = spaces.Box(
            low=-100.0, high=100.0, 
            shape=(10,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Dict(obs_dict)

    @property
    def action_dim(self):
        """兼容性属性：返回动作空间总维度。"""
        return sum(self.action_manager.action_term_dim)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Dict, Dict]:
        """Gymnasium 标准重置。"""
        # 遵循 Gymnasium 标准处理 seed (虽然目前 sim 不支持 seed)
        super().reset() # 调用 ManagerBasedEnv.reset
        
        # 获取初始观测并扁平化组 (目前只支持 policy 组)
        obs_raw = self.observation_manager.compute_observations()
        obs = self._process_obs(obs_raw)
        
        return obs, {}

    def step(self, action: torch.Tensor | np.ndarray) -> tuple[Dict, float, bool, bool, Dict]:
        """Gymnasium 标准步进。"""
        # 确保 action 是 Tensor
        if isinstance(action, (np.ndarray, list)):
            action = torch.as_tensor(action, device=self.device)
            
        # 调用 ManagerBasedEnv.step
        obs_raw, reward, terminated, truncated, info = super().step(action)
        
        # 处理观测
        obs = self._process_obs(obs_raw)
        
        # 转换为标量 (SB3 期望标量奖励)
        reward_scalar = float(reward.item())
        done = bool(terminated.item())
        trunc = bool(truncated.item())
        
        return obs, reward_scalar, done, trunc, info

    def _process_obs(self, obs_raw: Dict) -> Dict:
        """从 ObservationManager 的原始输出中提取并转换格式。"""
        # 目前主要关注 'policy' 组
        policy_group = obs_raw.get("policy", {})
        
        # 将 Tensor 转换为 Numpy (Gym 要求)
        processed = {}
        for k, v in policy_group.items():
            if isinstance(v, torch.Tensor):
                # 如果是单环境，移除 batch 维 (Gym 期望单实例观测)
                if v.shape[0] == 1:
                    v = v.squeeze(0)
                processed[k] = v.detach().cpu().numpy()
            else:
                processed[k] = v
        return processed
