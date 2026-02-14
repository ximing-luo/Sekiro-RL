from __future__ import annotations
import torch
import struct
from typing import TYPE_CHECKING, Sequence, Dict, Any

from ..asset_base import AssetBase
from .sekiro_asset_data import SekiroAssetData
from src.gamelab.interfaces.telemetry.driver import TelemetryDriver

if TYPE_CHECKING:
    from .sekiro_asset_cfg import SekiroAssetCfg

class SekiroAsset(AssetBase):
    """Sekiro 游戏资产实现。
    
    负责通过 TelemetryDriver 读取游戏遥测数据，并将数据填充到 SekiroAssetData (Tensors) 中。
    """
    
    def __init__(self, cfg: SekiroAssetCfg):
        super().__init__(cfg)
        self.cfg: SekiroAssetCfg = cfg
        self.base_address = None
        self.driver = TelemetryDriver(self.cfg.process_name)
        self._data = SekiroAssetData(num_envs=1, device=self.cfg.device)
        
        self._initialize_driver()

    def _initialize_driver(self):
        """连接进程并寻找特征码基址。违约即抛出异常。"""
        self.driver.connect()
        results = self.driver.pattern_scan_all(self.cfg.signature)
        if not results:
            raise RuntimeError(f"[SekiroAsset] 未找到遥测区签名 ({self.cfg.signature.hex()})。请检查游戏版本或签名配置。")
            
        self.base_address = results
        self._is_initialized = True
        print(f"[SekiroAsset] 找到遥测区签名，基址: {hex(self.base_address)}")

    @property
    def data(self) -> SekiroAssetData:
        return self._data

    def update(self, dt: float):
        """执行内存读取并更新 Tensor 缓冲区。
        
        基于“必然性”原则：如果未初始化，应在外部 setup 阶段拦截，而不是在此处卑微修补。
        """
        if not self._is_initialized: return

        # 0. 备份上一帧状态
        self._data.backup()

        # 1. 追求零成本抽象：批量读取原始字节并解包
        # 范围：从偏移 12 到 60 (共 48 字节)
        raw_data = self.driver.read_bytes(self.base_address + 12, 48)
        
        # 使用 struct.unpack 一次性解析所有字段 (I=uint32, 4x=4 bytes padding)
        values = struct.unpack("<II4xIIII4xIIII", raw_data)
        
        # 3. 直接同步到 Tensor 缓冲区，消除字典和重复读取开销
        self._data.update_from_raw(values)

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置资产状态。"""
        # 立即更新一次以获取最新状态
        self.update(0.0)
