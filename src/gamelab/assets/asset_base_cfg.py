from dataclasses import dataclass, MISSING
from typing import Type, Any

@dataclass
class AssetBaseCfg:
    """游戏资产的基础配置类。
    
    对标 Isaac Lab 的 AssetBaseCfg，用于定义资产的初始化参数。
    """
    
    name: str = "asset"
    """资产名称（如 'player', 'boss'）。"""

    num_envs: int = 1
    """资产所属的环境数量 (由 InteractiveScene 自动注入)。"""

    class_type: Any = None
    """关联的资产类。"""
    
    def validate(self):
        """校验配置有效性。"""
        if self.class_type is None:
            raise ValueError(f"资产配置 {self.name} 必须指定 class_type")
