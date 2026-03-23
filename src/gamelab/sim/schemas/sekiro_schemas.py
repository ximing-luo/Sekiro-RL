from dataclasses import dataclass

@dataclass
class SekiroCharacterSchemaCfg:
    """定义《只狼》角色的核心物理与逻辑属性模板。
    
    对标 Isaac Lab 的 schemas，用于规范资产的数值行为。
    """
    
    # 基础数值
    hp_max: float = 100.0
    posture_max: float = 100.0
    
    # 恢复速率
    posture_recovery_rate: float = 5.0  # 每秒恢复量
    hp_recovery_rate: float = 0.0
    
    # 动作判定帧 (示例)
    parry_window: float = 0.5  # 格挡有效窗口（秒）
    dodge_window: float = 0.3  # 闪避无敌帧（秒）
    
    # 碰撞/打击判定
    hitbox_radius: float = 0.5
    attack_range: float = 2.0
