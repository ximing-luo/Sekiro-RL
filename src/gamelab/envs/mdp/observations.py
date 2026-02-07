"""
模块用途：原子观测函数定义（Observation Terms）。
"""
import numpy as np
from src.gamelab.interfaces.observe.telemetry import SekiroTelemetry

# 初始化遥测单例
telemetry = SekiroTelemetry()
telemetry.start() # 启动后台刷新

def memory_metrics(env, **kwargs):
    """从内存读取的基础数值指标。"""
    metrics = telemetry.get_metrics()
    return {
        'self_blood': metrics["player_hp"],
        'self_blood_max': metrics["player_hp_max"],
        'boss_blood': metrics["enemy_hp"],
        'boss_blood_max': metrics["enemy_hp_max"],
        'self_stamina': metrics["player_posture"],
        'self_stamina_max': metrics["player_posture_max"],
        'boss_stamina': metrics["enemy_posture"],
        'boss_stamina_max': metrics["enemy_posture_max"],
        'player_deaths': metrics["player_deaths"],
        'enemy_deaths': metrics["enemy_deaths"]
    }

def image_frame(env, **kwargs):
    """获取最新图像帧并转换为 CHW 格式。"""
    if env.scene_manager:
        frame = env.scene_manager.get_latest_frame()
        if frame is not None:
            # OpenCV 默认是 HWC (H, W, 3)，转换为 PyTorch 要求的 CHW (3, H, W)
            if frame.ndim == 3 and frame.shape[-1] == 3:
                return frame.transpose(2, 0, 1)
        return frame
    return None

def last_action(env, action, **kwargs):
    """将上一步动作作为观测的一部分。"""
    return action
