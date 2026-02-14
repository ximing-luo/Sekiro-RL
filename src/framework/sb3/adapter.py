# Copyright (c) 2024, Sekiro-RL Project.
# All rights reserved.

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

class SB3VecEnvAdapter(VecEnv):
    """
    SB3 向量化环境适配器。
    
    负责将原生返回 PyTorch 张量的向量化环境包装为 SB3 兼容的 VecEnv。
    对标 Isaac Lab 的 Sb3VecEnvWrapper。
    """
    def __init__(self, env: Any, device: str = "cuda:0"):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        
        # 显式同步空间契约
        super().__init__(self.num_envs, env.observation_space, env.action_space)

    def reset(self) -> VecEnvObs:
        """重置环境并返回 NumPy 格式的观测。"""
        obs, _ = self.env.reset()
        return self._obs_to_numpy(obs)

    def step_async(self, actions: np.ndarray) -> None:
        """异步分发动作。"""
        # 将 NumPy 动作转换为环境预期的 Torch 张量
        self._actions = torch.from_numpy(actions).to(self.device)

    def step_wait(self) -> VecEnvStepReturn:
        """
        等待物理计算完成，并支付 GPU->CPU 的'过桥费'。
        注意：这是 SB3 架构的强制要求，虽然低效，但在不更换 RL 框架前无法避免。
        """
        # 1. 物理层同步
        obs, rewards, term, trunc, infos = self.env.step(self._actions)

        # 2. 核心数据搬运 (Batch Transfer)
        obs_np = self._obs_to_numpy(obs)
        rew_np = rewards.cpu().numpy()
        done_np = (term | trunc).cpu().numpy()

        # 3. Info 极简处理 (仅保留协议必须项，拒绝通用递归)
        info_list = [{} for _ in range(self.num_envs)]
        
        # 协议 A: 截断信号 (SB3 必须)
        if "time_out" in infos:
            trunc_np = trunc.cpu().numpy()
            for i, t in enumerate(trunc_np):
                if t:
                    info_list[i]["TimeLimit.truncated"] = True

        # 协议 B: 奖励分项 (调试必须)
        if "reward_components" in infos:
            # 批量搬运以减少 CUDA 同步开销
            rc_cpu = {k: v.cpu().numpy() for k, v in infos["reward_components"].items()}
            for i in range(self.num_envs):
                info_list[i]["reward_components"] = {k: v[i] for k, v in rc_cpu.items()}

        return obs_np, rew_np, done_np, info_list

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        return [getattr(self.env, attr_name)] * (len(indices) if indices else self.num_envs)

    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices: Optional[List[int]] = None, **method_kwargs: Any) -> List[Any]:
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs)] * (len(indices) if indices else self.num_envs)

    def _obs_to_numpy(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> VecEnvObs:
        """将 Torch 张量观测转换为 NumPy 数组。"""
        if isinstance(obs, dict):
            return {k: v.cpu().numpy() for k, v in obs.items()}
        return obs.cpu().numpy()

    def get_images(self) -> List[np.ndarray]:
        """返回环境渲染图（如果支持）。"""
        return [self.env.render()] * self.num_envs

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """设置随机种子。"""
        return [self.env.seed(seed)] * self.num_envs

    def env_is_wrapped(self, wrapper_class: Any, indices: Optional[List[int]] = None) -> List[bool]:
        """检查环境是否被特定包装器包装。"""
        return [False] * (len(indices) if indices else self.num_envs)
