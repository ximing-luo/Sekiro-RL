
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Type, Any
from src.gamelab.utils.configclass import configclass

@configclass
class AssetBaseCfg:
    """游戏资产的基础配置类。
    
    对标 Isaac Lab 的 AssetBaseCfg，用于定义资产的初始化参数。
    """
    
    class_type: Type[Any] = None
    """关联的资产类。"""
    
    name: str = MISSING
    """资产名称（如 'player', 'boss'）。"""
    
    def validate(self):
        """校验配置有效性。"""
        if self.class_type is None:
            raise ValueError(f"资产配置 {self.name} 必须指定 class_type")
