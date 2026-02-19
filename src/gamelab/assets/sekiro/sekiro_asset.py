from __future__ import annotations
import torch
import struct
from typing import TYPE_CHECKING, Sequence, Dict, Any, List

from ..asset_base import AssetBase
from .sekiro_asset_data import SekiroAssetData
from src.gamelab.interfaces.telemetry.driver import TelemetryDriver

if TYPE_CHECKING:
    from .sekiro_asset_cfg import SekiroAssetCfg

class SekiroAsset(AssetBase):
    """Sekiro 游戏资产实现。
    
    负责通过 TelemetryDriver 读取游戏遥测数据，并将数据填充到 SekiroAssetData (Tensors) 中。
    支持多环境并行读取 (Vectorized Asset)。
    """
    
    def __init__(self, cfg: SekiroAssetCfg):
        super().__init__(cfg)
        self.cfg: SekiroAssetCfg = cfg
        
        # 多环境支持
        self.drivers: List[TelemetryDriver] = []
        self.base_addresses: List[int] = []
        
        # 初始化数据容器
        # 注意：cfg.num_envs 已由 InteractiveScene 注入
        self._data = SekiroAssetData(num_envs=self.cfg.num_envs, device=self.cfg.device)
        
        self._initialize_drivers()
        print(f"[DEBUG][Asset] Initialized {self.cfg.num_envs} environments for {self.cfg.process_name}")

    def _initialize_drivers(self):
        """扫描并连接所有游戏进程。"""
        # 1. 查找所有 PID
        pids = TelemetryDriver.find_pids_by_name(self.cfg.process_name)
        required = self.cfg.num_envs
        # 确保只使用前 num_envs 个进程
        target_pids = pids[:required]
        if len(target_pids) < required: 
            raise RuntimeError(f"[SekiroAsset] 未找到足够的进程 {self.cfg.process_name} (需要 {required} 个，找到 {len(pids)} 个)")
        for i, pid in enumerate(target_pids):
            driver = TelemetryDriver(self.cfg.process_name)
            if not driver.connect(pid=pid):
                raise RuntimeError(f"环境 {i} 连接失败 (PID={pid})")
            self.drivers.append(driver)
            
            # 扫描特征码
            addr = driver.pattern_scan_all(self.cfg.signature)
            if not addr:
                raise RuntimeError(f"[SekiroAsset] 环境 {i} (PID={pid}) 未找到遥测区签名。")
            self.base_addresses.append(addr)
            print(f"[SekiroAsset] 环境 {i} (PID={pid}) 已连接，基址: {hex(addr)}")
            
        self._is_initialized = True

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

        # 1. 向量化读取：循环读取每个环境的数据并填入 Tensor
        # 虽然这里是 Python 循环，但读取内存是 IO 密集型，且 num_envs 通常不大 (<64)，
        # 相比于 RL 推理开销，这里的循环是可以接受的。
        for i, (driver, base_addr) in enumerate(zip(self.drivers, self.base_addresses)):
            try:
                # 范围：从偏移 12 到 60 (共 48 字节)
                raw_data = driver.read_bytes(base_addr + 12, 48)
                # 使用 struct.unpack 一次性解析所有字段 (I=uint32, 4x=4 bytes padding)
                values = struct.unpack("<II4xIIII4xIIII", raw_data)
                # 3. 直接同步到 Tensor 缓冲区
                self._data.update_from_raw(values, env_id=i)
            except Exception as e:
                # 容错处理：读取失败不应导致训练崩溃，可以是警告
                # print(f"[SekiroAsset] 环境 {i} 读取警告: {e}")
                pass

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置资产状态。"""
        # 立即更新一次以获取最新状态
        self.reset_state()
        self.update(0.0)

    def reset_state(self):
        """重置敌我状态。"""
        for i, (driver, base_addr) in enumerate(zip(self.drivers, self.base_addresses)):
            driver.write_int(base_addr + 60, 1)
            
