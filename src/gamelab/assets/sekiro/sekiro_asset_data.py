
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

@dataclass
class SekiroAssetData:
    """Sekiro 资产的数据容器。
    
    存储玩家和敌人的实时状态（HP, Posture 等），全部采用 Tensor 格式。
    支持记录上一帧状态，以便计算奖励和检测事件。
    """
    
    def __init__(self, num_envs: int, device: str):
        self.device = device
        self.num_envs = num_envs
        
        # 当前状态
        self.player_hp = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.player_hp_max = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.player_posture = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.player_posture_max = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.enemy_hp = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.enemy_hp_max = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.enemy_posture = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.enemy_posture_max = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.player_deaths = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.enemy_deaths = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        # 上一帧状态 (用于计算 Delta)
        self.prev_player_hp = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.prev_player_posture = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.prev_enemy_hp = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.prev_enemy_posture = torch.zeros(num_envs, dtype=torch.float32, device=device)

    def update_prev(self):
        """将当前状态备份到上一帧。"""
        self.prev_player_hp.copy_(self.player_hp)
        self.prev_player_posture.copy_(self.player_posture)
        self.prev_enemy_hp.copy_(self.enemy_hp)
        self.prev_enemy_posture.copy_(self.enemy_posture)

    def update_from_dict(self, data_dict: dict, env_id: int = 0):
        """将从内存读取的字典数据同步到 Tensor 缓冲区。"""
        self.player_hp[env_id] = float(data_dict.get("player_hp", 0))
        self.player_hp_max[env_id] = float(data_dict.get("player_hp_max", 1))
        self.player_posture[env_id] = float(data_dict.get("player_posture", 0))
        self.player_posture_max[env_id] = float(data_dict.get("player_posture_max", 1))
        
        self.enemy_hp[env_id] = float(data_dict.get("enemy_hp", 0))
        self.enemy_hp_max[env_id] = float(data_dict.get("enemy_hp_max", 1))
        self.enemy_posture[env_id] = float(data_dict.get("enemy_posture", 0))
        self.enemy_posture_max[env_id] = float(data_dict.get("enemy_posture_max", 1))
        
        self.player_deaths[env_id] = int(data_dict.get("player_deaths", 0))
        self.enemy_deaths[env_id] = int(data_dict.get("enemy_deaths", 0))
