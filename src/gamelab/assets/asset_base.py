from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .asset_base_cfg import AssetBaseCfg

class AssetBase(ABC):
    """游戏资产的基础接口类。
    
    对标 Isaac Lab 的 AssetBase。在游戏强化学习语境下，资产代表游戏中的实体（如玩家、敌人）。
    其核心功能是维护一个 Tensor 缓冲区（AssetData），并将其与游戏内存数据同步。
    """
    
    def __init__(self, cfg: AssetBaseCfg):
        """初始化资产。
        
        Args:
            cfg: 资产配置类。
        """
        self.cfg = cfg
        self._is_initialized = False
        self._device = "cpu"
        self._num_instances = 1
        
    @property
    def is_initialized(self) -> bool:
        """资产是否已初始化。"""
        return self._is_initialized
    
    @property
    def num_instances(self) -> int:
        """资产实例数量。"""
        return self._num_instances
    
    @property
    def device(self) -> str:
        """计算设备。"""
        return self._device
    
    @property
    @abstractmethod
    def data(self) -> Any:
        """资产相关数据（AssetData）。"""
        pass
    
    @abstractmethod
    def update(self, dt: float):
        """从游戏内存更新资产数据。
        
        Args:
            dt: 时间步长。
        """
        pass
    
    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None):
        """重置选定环境的内部缓冲区。
        
        Args:
            env_ids: 环境 ID 序列。如果为 None，则重置所有环境。
        """
        pass

    def __getitem__(self, env_id: int) -> Any:
        """获取指定环境的资产数据。（如 asset[env_id]）
        
        Args:
            env_id: 环境 ID。
        
        Returns:
            资产数据。
        """
        return self.data[env_id]

    def __getattr__(self, name: str) -> Any:
        """获取资产数据的属性。(如 asset.player)
        
        Args:
            name: 属性名。
        
        Returns:
            属性值。
        """
        return getattr(self.data, name)
