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
        # A. 动作空间 (目前支持 MultiDiscrete)
        # TODO: 后续可根据 ActionManager 的具体 Term 类型扩展为 Discrete 或 Box
        self.action_space = spaces.MultiDiscrete(self.action_manager.action_term_dim)
        
        # B. 观测空间 (通过试运行 compute_observations 动态推导维度)
        # 这样即使在配置中增删观测项，也不需要手动修改这里的 Box 定义
        with torch.no_grad():
            # 确保资产已经初始化，以便获取正确的 shape
            self.scene.update(0.0)
            obs_raw = self.observation_manager.compute_observations()
            
        # 目前只处理 policy 组，如果需要支持多组观测，可在此扩展
        policy_group = obs_raw.get("policy", {})
        if not policy_group:
            raise ValueError("[ManagerBasedRLEnv] 观测配置中未找到 'policy' 组，无法配置观测空间。")

        obs_dict = {}
        for key, value in policy_group.items():
            # 忽略 batch 维度 [num_envs, ...] -> [...]
            shape = value.shape[1:]
            
            # 针对不同类型的观测设置合理的上下限
            if key == "image":
                low, high = 0.0, 1.0
            elif key == "Telemetry":
                # 遥测数据等通用项使用无界限，防止硬编码 100 导致的溢出或截断
                low, high = -np.inf, np.inf
                
            obs_dict[key] = spaces.Box(
                low=low, high=high, 
                shape=shape, 
                dtype=np.float32
            )
            
        self.observation_space = spaces.Dict(obs_dict)
        print(f"[ManagerBasedRLEnv] 动态配置观测空间完成: {list(obs_dict.keys())}")

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
