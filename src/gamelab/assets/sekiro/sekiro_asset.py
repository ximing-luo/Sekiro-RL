from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Sequence, Dict, Any

from ..asset_base import AssetBase
from .sekiro_asset_data import SekiroAssetData
from src.gamelab.interfaces.memory.driver import MemoryDriver

if TYPE_CHECKING:
    from .sekiro_asset_cfg import SekiroAssetCfg

class SekiroAsset(AssetBase):
    """Sekiro 游戏资产实现。
    
    负责通过 MemoryDriver 读取游戏内存，并将数据填充到 SekiroAssetData (Tensors) 中。
    """
    
    def __init__(self, cfg: SekiroAssetCfg):
        super().__init__(cfg)
        self.cfg: SekiroAssetCfg = cfg
        
        # 初始化驱动
        self.driver = MemoryDriver(self.cfg.process_name)
        self.base_address = None
        
        # 初始化数据容器 (目前固定为 1 个环境，支持未来扩展)
        self._data = SekiroAssetData(num_envs=1, device=self.cfg.device)
        
        self._initialize_driver()

    def _initialize_driver(self):
        """连接进程并寻找特征码基址。"""
        if self.driver.connect():
            results = self.driver.pattern_scan_all(self.cfg.signature)
            if results:
                self.base_address = results
                self._is_initialized = True
                print(f"[SekiroAsset] 找到遥测区签名，基址: {hex(self.base_address)}")
            else:
                print("[SekiroAsset] 警告：未找到遥测区签名！")

    @property
    def data(self) -> SekiroAssetData:
        return self._data

    def _read_r32(self, offset: int) -> int:
        if self.base_address:
            return self.driver.read_int(self.base_address + offset)
        return 0

    def update(self, dt: float):
        """执行内存读取并更新 Tensor 缓冲区。"""
        if not self._is_initialized or not self.driver.pm:
            self._initialize_driver()
            if not self._is_initialized:
                return

        # 0. 备份上一帧状态
        self._data.update_prev()

        # 1. 批量读取原始数据 (偏移对标 telemetry.py)
        raw_values = {
            "player_hp": self._read_r32(12),
            "player_hp_max": self._read_r32(16),
            "player_posture": self._read_r32(24),
            "player_posture_max": self._read_r32(28),
            "enemy_hp": self._read_r32(32),
            "enemy_hp_max": self._read_r32(36),
            "enemy_posture": self._read_r32(44),
            "enemy_posture_max": self._read_r32(48),
            "player_deaths": self._read_r32(52),
            "enemy_deaths": self._read_r32(56),
        }

        # 2. 同步到 Tensor 缓冲区
        self._data.update_from_dict(raw_values)

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置资产状态。"""
        # 立即更新一次以获取最新状态
        self.update(0.0)
