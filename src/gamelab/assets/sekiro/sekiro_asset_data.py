import torch
from dataclasses import dataclass
from typing import List

class StateBuffer:
    """双缓冲状态容器：自动化管理当前值与上一帧值。"""
    def __init__(self, num_envs: int, device: str, fields: List[str]):
        self.fields = fields
        dtype = torch.int32
        
        # 存储张量字典，便于循环操作
        self.storage = {f: torch.zeros(num_envs, dtype=dtype, device=device) for f in fields}
        self.prev_storage = {f: torch.zeros(num_envs, dtype=dtype, device=device) for f in fields}
        
        # 动态绑定属性，支持 .hp 这种直接访问方式
        for field in fields:
            setattr(self, field, self.storage[field])
            setattr(self, f"prev_{field}", self.prev_storage[field])

    def backup(self):
        """一键备份当前状态到 prev_ 缓冲区。"""
        for f in self.fields:
            self.prev_storage[f].copy_(self.storage[f])

    def update(self, values: tuple, env_id: int = 0):
        """按顺序将原始数值填充到张量缓冲区。"""
        for i, f in enumerate(self.fields):
            self.storage[f][env_id] = values[i]

@dataclass
class SekiroAssetData:
    """Sekiro 资产的数据容器。
    
    采用“状态组”结构，实现自动化双缓冲管理。
    """
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        
        # 1. 定义状态组：配置即逻辑
        self.player = StateBuffer(num_envs, device, ["hp", "hp_max", "posture", "posture_max"])
        self.enemy = StateBuffer(num_envs, device, ["hp", "hp_max", "posture", "posture_max"])
        self.status = StateBuffer(num_envs, device, ["player_deaths", "enemy_deaths"])

    def backup(self):
        """同步备份所有状态组。"""
        self.player.backup()
        self.enemy.backup()
        self.status.backup()

    def update_from_raw(self, raw_data: tuple, env_id: int = 0):
        """将 TelemetryDriver 读出的原始元组分发给各个状态组。"""
        # 依据内存布局顺序切片分发
        self.player.update(raw_data[0:4], env_id)
        self.enemy.update(raw_data[4:8], env_id)
        self.status.update(raw_data[8:10], env_id)
