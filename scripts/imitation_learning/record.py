"""
Sekiro-RL 模仿学习录制脚本 (Imitation Learning Recording)
使用人类操作录制素材，数据由 RecorderManager 自动保存为高性能 .npy 格式。

使用方法:
1. 确保 OBS 已启动“虚拟摄像头”。
2. 运行脚本: python scripts/imitation_learning/record.py
3. 按 F9 开始录制，再按 F9 停止并保存。
4. 按 F8 退出。
"""

import os
import sys
import torch
import numpy as np
import win32api as wapi
import win32con
import time
from datetime import datetime

# 加入项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gamelab.envs.manager_based_rl_env import ManagerBasedRLEnv
from src.tasks.sekiro.sekiro_env_cfg import SekiroEnvCfg
from src.gamelab.interfaces.input.monitor import InputMonitor

# 映射 11 位按键到索引
ACTION_KEY_MAP = {
    'W': 0, 'A': 1, 'S': 2, 'D': 3,
    'SPACE': 4, 'J': 5, 'M_LEFT': 5,  # 鼠标左键也映射到攻击
    'G': 6, 'Q': 7,
    'SHIFT': 8, 'R': 9, 'X': 10
}

def get_human_action(monitor, num_envs, device):
    """获取人类按键和鼠标位移，转化为 13 维动作 Tensor。"""
    keys, dx, dy = monitor.get_input()
    
    # 1. 填充 11 个按键位
    mask = np.zeros(11, dtype=np.float32)
    for k in keys:
        if k in ACTION_KEY_MAP:
            mask[ACTION_KEY_MAP[k]] = 1.0
            
    # 2. 合并鼠标 dx, dy (作为第 11, 12 位)
    # 将 dx, dy 限制在合理的范围并转化为 float32
    action_vec = np.concatenate([mask, [float(dx), float(dy)]])
    
    # 3. 扩展为 batch (num_envs, 13) 并移动到设备
    action_tensor = torch.from_numpy(np.tile(action_vec, (num_envs, 1))).to(device)
    return action_tensor

def main():
    # 1. 初始化环境配置
    env_cfg = SekiroEnvCfg()
    # 确保只有一个环境进行录制，方便人机交互
    # env_cfg.scene.num_envs = 1 
    env_cfg.device: str = "cuda:0"
    env_cfg.headless: bool = False
    
    # 控制录制帧率 (FPS)
    target_fps = 30 
    env_cfg.scene.capture_fps = target_fps # 设置仿真步长 dt = 1/target_fps
    max_steps = 100000 # 录制最大步数 (约 100秒 @ 30FPS)
    frame_time = 1.0 / target_fps
    
    # 2. 实例化环境与监测器
    env = ManagerBasedRLEnv(env_cfg)
    # 初始化监测器，暂不开启回正（按F9录制时才动态开启更安全，但这里先常驻开启简化逻辑）
    monitor = InputMonitor()
    
    print("\n" + "="*55)
    print("Sekiro-RL 模仿学习录制工具 (人机耦合模式)")
    print("="*55)
    print(f"分辨率: {env_cfg.scene.observation_w}x{env_cfg.scene.observation_h} (C, H, W)")
    print(f"动作空间: {env.action_space}")
    print(f"存储路径: outputs/recordings/")
    print("-"*55)
    print("【F9】: 开启/停止录制")
    print("【F8】: 退出脚本")
    print("="*55 + "\n")
    
    recording = False
    f9_pressed = False
    start_step = 0
    
    # 第一次重置以获取初始观测
    env.reset()
    
    try:
        while True:
            loop_start = time.time()
            # 检查 F8 退出
            if wapi.GetAsyncKeyState(win32con.VK_F8) & 0x8000:
                print("\n[退出] 正在关闭环境...")
                break
            
            # 检查 F9 切换录制状态 (防抖处理)
            f9_state = wapi.GetAsyncKeyState(win32con.VK_F9) & 0x8000
            if f9_state and not f9_pressed:
                recording = not recording
                f9_pressed = True
                
                if recording:
                    status = f">>> 开始录制... (目标: {max_steps} 帧 | FPS: {target_fps})"
                    start_step = env.sim.step_count
                else:
                    status = "<<< 停止录制并保存"
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
                
                # 动态控制 RecorderManager
                if env.recorder_manager:
                    env.recorder_manager.set_recording_enabled(range(env.num_envs), recording)
            elif not f9_state:
                f9_pressed = False
            
            # 自动结束录制逻辑
            if recording:
                current_steps = env.sim.step_count - start_step
                if current_steps >= max_steps:
                    recording = False
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] <<< 已达到最大帧数 {max_steps}，自动停止录制并保存")
                    if env.recorder_manager:
                        env.recorder_manager.set_recording_enabled(range(env.num_envs), False)
            
            # 获取人类输入作为动作
            action = get_human_action(monitor, env.num_envs, env.device)
            
            # 步进环境 (RecorderManager 会在内部 step 中自动记录)
            obs, reward, done, time_out, info = env.step(action)
            
            if recording:
                print(f"\r录制进度: {env.sim.step_count - start_step}/{max_steps}", end="", flush=True)

            # 控制录制频率
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            
    except KeyboardInterrupt:
        print("\n用户手动中断。")
    except Exception as e:
        print(f"\n运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.reset()
        env.close()
        print("环境已关闭。")

if __name__ == "__main__":
    main()
