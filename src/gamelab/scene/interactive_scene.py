from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Dict, Sequence

if TYPE_CHECKING:
    from src.gamelab.envs.manager_based_env_cfg import SceneCfg
    from src.gamelab.assets.asset_base import AssetBase

class InteractiveScene:
    """交互式场景类，负责管理和同步场景中的所有资产。
    
    对标 Isaac Lab 的 InteractiveScene。
    """
    
    def __init__(self, cfg: SceneCfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.assets: Dict[str, AssetBase] = {}
        
        # 初始化资产
        for name, asset_cfg in self.cfg.assets.items():
            asset_cfg.name = name
            asset_cfg.device = device
            asset_cfg.validate()
            self.assets[name] = asset_cfg.class_type(asset_cfg)
            
    def update(self, dt: float):
        """同步场景中所有资产的状态。"""
        for asset in self.assets.values():
            asset.update(dt)
            
    def reset(self, env_ids: Sequence[int] | None = None):
        """重置场景中的资产。"""
        for asset in self.assets.values():
            asset.reset(env_ids)

    def write_data_to_sim(self):
        """将 Python 层资产数据写入仿真器（对标 Isaac Lab）。
        
        目前《只狼》任务主要通过 Telemetry 读取内存，暂无反向写入需求。
        """
        pass

    def __getitem__(self, key: str) -> AssetBase:
        """获取指定名称的资产。"""
        return self.assets[key]

    def __getattr__(self, name: str) -> AssetBase:
        """允许通过属性访问资产（如 scene.player）。"""
        if name in self.assets:
            return self.assets[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
