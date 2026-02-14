"""
只狼特定的原子奖励函数。
这些函数尽量复用 mdp.common 中的通用逻辑，仅保留只狼特有的业务判断。
数值权重已移至 Config 层。
"""

import torch as th
from typing import Dict, List, Any, TYPE_CHECKING
from src.gamelab.managers.reward_manager import RewardTerm, DeltaRewardTerm, EventRewardTerm
from src.gamelab.managers.manager_term_cfg import RewardTermCfg, DeltaRewardTermCfg, EventRewardTermCfg

class SekiroSurvivalRewardTerm(RewardTerm):
    """只狼生存/动作成本奖励。"""
    def __call__(self) -> th.Tensor:
        move_cost = self.cfg.params.get("move_cost", -0.05)
        skill_cost = self.cfg.params.get("skill_cost", -0.02)
        
        # 从 ActionManager 获取最新动作
        action = self._env.action_manager.action
        
        # 针对 MultiDiscrete 动作 [num_envs, action_dim]
        # action[:, 0] 是移动，action[:, 1] 是技能
        move_indices = action[:, 0]
        skill_indices = action[:, 1]
        
        rewards = th.zeros(self.num_envs, device=self.device)
        rewards[move_indices != 0] += move_cost
        rewards[skill_indices != 0] += skill_cost
        
        return rewards

def delta_reward_cfg(key: str, weight: float = 1.0, scale: float = 1.0) -> DeltaRewardTermCfg:
    """增量奖励配置。"""
    # 追求强契约：使用专门的 DeltaRewardTermCfg
    return DeltaRewardTermCfg(
        class_type=DeltaRewardTerm,
        weight=weight,
        group_name="telemetry",
        term_name=key,
        scale=scale
    )

def event_reward_cfg(event_id: int, reward: float = 1.0, weight: float = 1.0) -> EventRewardTermCfg:
    """事件奖励配置。"""
    return EventRewardTermCfg(
        class_type=EventRewardTerm,
        weight=weight,
        event_id=event_id,
        reward=reward
    )
