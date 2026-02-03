import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from src.model import SekiroMLPExtractor

class GroupLRWrapper(dict):
    """包装参数组字典，拦截 'lr' 设置并应用缩放比例"""
    def __init__(self, group, multiplier):
        super().__init__(group)
        self.multiplier = multiplier
    
    def __setitem__(self, key, value):
        if key == 'lr':
            # 当 SB3 尝试设置 lr 时，乘以预设的比例
            super().__setitem__(key, value * self.multiplier)
        else:
            super().__setitem__(key, value)

class SekiroCustomPolicy(ActorCriticPolicy):
    """自定义策略类，支持为特征提取器设置独立学习率并使用带归一化的 MLP 头部"""
    
    def _build_mlp_extractor(self) -> None:
        """重写以使用带 RMSNorm 的 SekiroMLPExtractor"""
        self.mlp_extractor = SekiroMLPExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _make_optimizer(self) -> torch.optim.Optimizer:
        # 定义参数组及其缩放比例
        # 让特征提取器（眼睛）的学习率始终是总学习率的 0.1 倍
        param_groups = [
            {"params": self.features_extractor.parameters(), "lr_multiplier": 0.1},
            {"params": self.mlp_extractor.parameters(), "lr_multiplier": 1.0},
            {"params": self.action_net.parameters(), "lr_multiplier": 1.0},
            {"params": self.value_net.parameters(), "lr_multiplier": 1.0},
        ]
        
        # 使用基础学习率初始化优化器
        base_lr = self.lr_schedule(1)
        optimizer = torch.optim.Adam(param_groups, lr=base_lr, **self.optimizer_kwargs)
        
        # 包装 param_groups 以拦截后续的自动更新
        new_groups = []
        for group in optimizer.param_groups:
            multiplier = group.pop("lr_multiplier", 1.0)
            new_groups.append(GroupLRWrapper(group, multiplier))
        
        optimizer.param_groups = new_groups
        return optimizer
