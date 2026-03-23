"""
模块用途：原子观测函数定义（Observation Terms）。
"""
import torch
import numpy as np

# 遥测指标的键名列表，用于映射 Tensor 索引
TELEMETRY_KEYS = [
    'player_hp', 'player_hp_max', 'player_posture', 'player_posture_max',
    'enemy_hp', 'enemy_hp_max', 'enemy_posture', 'enemy_posture_max',
    'player_deaths', 'enemy_deaths'
]

def memory_metrics(env, **kwargs):
    """从内存读取的基础数值指标。对标 Isaac Lab 的 Telemetry Injection。"""
    # 追求强契约：直接从环境绑定的资产获取数据，消除字典查找
    data = env.scene.sekiro.data
    
    # 返回归一化后的指标张量 [num_envs, 10]
    # 拼接顺序：HP, HP_Max, Posture, Posture_Max, Boss_HP, Boss_HP_Max, Boss_Posture, Boss_Posture_Max, Player_Deaths, Enemy_Deaths
    metrics = torch.stack([
        data.player.hp,
        data.player.hp_max,
        data.player.posture,
        data.player.posture_max,
        data.enemy.hp,
        data.enemy.hp_max,
        data.enemy.posture,
        data.enemy.posture_max,
        data.status.player_deaths,
        data.status.enemy_deaths
    ], dim=-1)
    
    return metrics

def image_frame(env, **kwargs):
    """获取最新图像帧并转换为 CHW 格式的 Tensor。"""
    sensor = env.sim.get_sensor("vision")
    num_envs = env.num_envs
    
    if sensor:
        frame = sensor.get_data()
        if frame is not None:
            # OpenCV 默认是 HWC (H, W, 3)，转换为 PyTorch 要求的 CHW (3, H, W)
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = frame.transpose(2, 0, 1)
            
            # 转换为 Tensor 并归一化 [3, H, W]
            frame_tensor = torch.from_numpy(frame).to(env.device)
            # 扩展为 [num_envs, 3, H, W]
            return frame_tensor.unsqueeze(0).repeat(num_envs, 1, 1, 1)
            
    return torch.zeros((num_envs, 3, env.cfg.scene.observation_h, env.cfg.scene.observation_w), device=env.device)

def last_action(env, **kwargs):
    """将上一步动作作为观测的一部分。"""
    return env.action_manager.action
