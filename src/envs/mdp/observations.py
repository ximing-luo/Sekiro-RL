"""
模块用途：原子观测函数定义（Observation Terms）。
"""
import numpy as np
from src.interfaces.observe.telemetry import SekiroTelemetry

# 初始化遥测单例
telemetry = SekiroTelemetry()
telemetry.start() # 启动后台刷新

def memory_metrics(env, **kwargs):
    """从内存读取的基础数值指标。"""
    sb, bb, ss, bs, bss = telemetry.get_metrics()
    return {
        'self_blood': sb,
        'boss_blood': bb,
        'self_stamina': ss,
        'boss_stamina': bs,
        'boss_stamina_max': bss
    }

def image_frame(env, **kwargs):
    """获取最新图像帧。"""
    if env.scene_manager:
        return env.scene_manager.get_latest_frame()
    return None

def last_action(env, action, **kwargs):
    """将上一步动作作为观测的一部分。"""
    return action
