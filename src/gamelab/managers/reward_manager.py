from __future__ import annotations
import torch
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Any, Sequence, Tuple
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RewardTermCfg, DeltaRewardTermCfg, EventRewardTermCfg

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env import ManagerBasedEnv

class RewardTerm(ManagerTermBase):
    """奖励术语基类。"""
    @abstractmethod
    def __call__(self) -> torch.Tensor:
        """计算奖励值。"""
        raise NotImplementedError

class StandardRewardTerm(RewardTerm):
    """标准奖励术语（包装旧的函数式逻辑）。"""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._device_obj = torch.device(self.device)

    def __call__(self) -> torch.Tensor:
        # 追求零成本抽象：不再显式传递 metrics，让函数通过 env 自行获取
        return self.cfg.func(
            env=self._env,
            **self.cfg.params
        )

class DeltaRewardTerm(RewardTerm):
    """增量奖励术语。
    
    根据资产数据中“当前值”与“上一帧值”的变化（Delta）计算奖励。
    """
    def __call__(self) -> torch.Tensor:
        cfg: DeltaRewardTermCfg = self.cfg
        asset_name = cfg.group_name # 在 Sekiro 场景中，group_name 通常对应资产名，如 "player"
        field_name = cfg.term_name   # 对应字段名，如 "hp"
        
        # 1. 获取资产数据对象
        asset = self._env.scene.assets.get(asset_name)
        if asset is None:
            return torch.zeros(self.num_envs, device=self.device)
            
        data = asset.data
        
        # 2. 通过属性映射获取当前值和上一帧值 (利用 sekiro_asset_data 的动态属性)
        # 例如: data.player.hp 和 data.player.prev_hp
        # 注意：sekiro_asset_data 内部结构是 data.player.hp
        # 如果 cfg.group_name 是 "player", cfg.term_name 是 "hp"
        state_group = getattr(data, asset_name) 
        current_val = getattr(state_group, field_name)
        prev_val = getattr(state_group, f"prev_{field_name}")
        
        # 3. 计算变化量并缩放
        delta = current_val - prev_val
        return delta * cfg.scale

class EventRewardTerm(RewardTerm):
    """事件奖励术语。
    
    当特定事件发生时给予奖励。由于事件是由 EventManager 生成的，
    此术语需要访问 env.event_manager 的最新结果。
    """
    def __call__(self) -> torch.Tensor:
        cfg: EventRewardTermCfg = self.cfg
        event_id = cfg.event_id
        
        # 向量化处理：创建一个 (num_envs,) 的零张量
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 从环境获取最新触发的事件
        # 注意：此处需要环境或 EventManager 提供最近一帧触发事件的接口
        recent_events = self._env.event_manager.recent_events
        
        if event_id in recent_events:
            rewards.fill_(cfg.reward)
            
        return rewards

class RewardManager(ManagerBase):
    """奖励管理器：实现基于术语（Term-based）的奖励计算。
    """
    _TERM_CLASS = StandardRewardTerm

    def __init__(self, cfg: Dict[str, RewardTermCfg], env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._reward_buf = torch.zeros(self.num_envs, device=self.device)

    def step(self, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算并返回总奖励及分量。对标 Isaac Lab。"""
        self._reward_buf.zero_()
        components = {}
        
        for name in self._term_names:
            term: RewardTerm = self._terms[name]
            term_cfg = self.cfg[name]
            
            # 计算原始值 (不再传递 metrics)
            val = term()
                
            weighted_val = val * term_cfg.weight * dt # 遵循 Isaac Lab 乘以 dt 的规范
            self._reward_buf += weighted_val
            
            components[name] = weighted_val
            
        return self._reward_buf.clone(), components
