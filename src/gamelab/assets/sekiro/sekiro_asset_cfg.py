from dataclasses import dataclass, field
from typing import Dict
from ..asset_base_cfg import AssetBaseCfg
from src.gamelab.utils.configclass import configclass
from . import keys

@configclass
class SekiroAssetCfg(AssetBaseCfg):
    """Sekiro 资产的配置类。"""
    
    from .sekiro_asset import SekiroAsset
    class_type: type = SekiroAsset
    
    process_name: str = "sekiro.exe"
    """游戏进程名称。"""
    
    signature: bytes = b"SEKIRO_TLM"
    """内存遥测区的特征码。"""
    
    update_rate: float = 100.0
    """更新频率 (Hz)。"""
    
    key_mappings: Dict[str, int] = field(default_factory=lambda: {
        "W": keys.W,
        "A": keys.A,
        "S": keys.S,
        "D": keys.D,
        "M": keys.M,
        "J": keys.J,
        "K": keys.K,
        "LSHIFT": keys.LSHIFT,
        "R": keys.R,
        "V": keys.V,
        "Q": keys.Q,
        "SPACE": keys.SPACE,
    })
    """按键名称到扫描码的映射。"""
