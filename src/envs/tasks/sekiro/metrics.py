"""
模块用途：在环境中统一提取当前帧的血量与架势等指标。

包含：
- 函数：extract_metrics(observe, blood_window, stamina_window)

边界：
- 负责：调用 image_process 完成裁剪与数值提取
- 不负责：奖励计算、采集流程与训练逻辑
"""
import os
import sys

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import src.interfaces.observe.processor as image_process
from src.interfaces.observe.telemetry import SekiroTelemetry

# 初始化遥测单例
telemetry = SekiroTelemetry()
telemetry.start() # 启动后台刷新

# 从内存获取血量与架势等指标
def extract_metrics_from_memory():
    """
    直接从游戏内存读取指标，不再依赖图像处理。
    返回：自身血量, Boss血量, 自身架势, Boss架势
    """
    return telemetry.get_metrics()

# 从当前帧提取血量与架势等指标（保留原图像处理接口供参考或备用）
# 参数：
#   observe: 当前帧图像
#   blood_window: 血量区域窗口坐标
#   stamina_window: 架势区域窗口坐标
# 返回：
#   next_self_blood: 自身血量
#   next_boss_blood: Boss血量
#   next_self_stamina: 自身架势
#   next_boss_stamina: Boss架势
def extract_metrics(observe, blood_window, stamina_window):
    blood_window_color = image_process.crop_image(observe, [blood_window])[0]
    stamina_window_color = image_process.crop_image(observe, [stamina_window])[0]
    next_self_blood = image_process.self_blood_count(blood_window_color)
    next_boss_blood = image_process.boss_blood_count(blood_window_color)
    next_self_stamina = image_process.self_stamina_count(stamina_window_color)
    next_boss_stamina = image_process.boss_stamina_count(stamina_window_color)
    return next_self_blood, next_boss_blood, next_self_stamina, next_boss_stamina
