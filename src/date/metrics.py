"""
模块用途：在环境中统一提取当前帧的血量与架势等指标。

包含：
- 函数：extract_metrics(observe, blood_window, stamina_window)

边界：
- 负责：调用 image_process 完成裁剪与数值提取
- 不负责：奖励计算、采集流程与训练逻辑
"""
import numpy as np
import date.image_process as image_process

def extract_metrics(observe, blood_window, stamina_window):
    blood_window_color = image_process.crop_image(observe, [blood_window])[0]
    stamina_window_color = image_process.crop_image(observe, [stamina_window])[0]
    next_self_blood = image_process.self_blood_count(blood_window_color)
    next_boss_blood = image_process.boss_blood_count(blood_window_color)
    next_self_stamina = image_process.self_stamina_count(stamina_window_color)
    next_boss_stamina = image_process.boss_stamina_count(stamina_window_color)
    return next_self_blood, next_boss_blood, next_self_stamina, next_boss_stamina