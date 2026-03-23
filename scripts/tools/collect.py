"""
Sekiro-RL 采集工具 (Data Collector)
启动环境并通过随机动作采集真实的观测数据（obs），用于 benchmark。

使用方法:
python scripts/tools/collect.py
"""

import os
import sys
import torch
import numpy as np
import time

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tasks.registration import task_registry
import src.tasks.sekiro  # 触发注册

def collect_data(task_id="Sekiro-v0", num_frames=512*20, fps=30):
    """
    启动只狼环境，通过随机动作采集真实观测数据并保存。
    """
    print(f"\n" + "="*50)
    print(f"正在启动环境以采集数据: {task_id} | 目标速度: {fps} FPS")
    print("="*50)
    
    try:
        # 1. 获取并修改配置
        env_cfg = task_registry.get_task_cfg(task_id)
        
        # 注入运行时必要的参数 (模拟 AppLauncher 的行为)
        env_cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        env_cfg.headless = False
        
        # 修改采集帧率
        if hasattr(env_cfg, 'scene'):
            env_cfg.scene.capture_fps = fps
            env_cfg.scene.num_envs = 1
            
        # 2. 创建环境
        env = task_registry.make(task_id, cfg=env_cfg)
        
        obs, info = env.reset()
        all_obs = []
        
        print(f"\n准备就绪！开始采集 {num_frames} 帧数据...")
        start_time = time.time()
        interval = 1.0 / fps
        
        for i in range(num_frames):
            loop_start = time.time()
            
            # 随机动作以获得多样化画面
            action = env.action_space.sample()
            
            # 如果是单环境，sample() 返回的是单个动作，需要升维成 (1, D)
            # 根据错误提示 array is 1-dimensional, but 2 were indexed，说明 action 是 (D,) 而 ActionManager 期望 (N, D)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, device=env_cfg.device)
            
            if action.dim() == 1:
                action = action.unsqueeze(0)
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 处理字典观测
            if isinstance(obs, dict):
                # 优先获取 visual/policy 观测
                if "policy" in obs:
                    frame = obs["policy"]
                elif "visual" in obs:
                    frame = obs["visual"]
                else:
                    # 尝试取第一个 value
                    frame = next(iter(obs.values()))
            else:
                frame = obs
                
            # 确保 frame 是 numpy 数组 (env.step 可能返回 tensor 或 numpy)
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            
            # 如果是单环境，移除 batch 维度 (1, C, H, W) -> (C, H, W)
            if frame.shape[0] == 1:
                frame = frame.squeeze(0)
            
            # 关键修复：确保数据独立拷贝，防止引用导致所有帧都变成最后一帧
            frame = frame.copy()
                
            # obs 期望是 (C, H, W) 的 uint8 数组
            all_obs.append(frame)
            
            if done:
                obs, info = env.reset()
                
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                current_fps = (i + 1) / elapsed
                print(f"进度: {i+1}/{num_frames} | 实际速度: {current_fps:.2f} FPS")
            
            # 控制采集速度
            elapsed_loop = time.time() - loop_start
            sleep_time = interval - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        env.close()
        
        # 转换为 numpy 数组并保存
        all_obs = np.array(all_obs, dtype=np.uint8)
        
        save_dir = os.path.join(project_root, "logs", "data")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, "benchmark_obs.npy")
        np.save(save_path, all_obs)
        
        print("\n" + "="*50)
        print(f"采集完成！")
        print(f"数据保存至: {save_path}")
        print(f"数据形状: {all_obs.shape}")
        
        # 简单校验数据是否有效
        if len(all_obs) > 1:
            # 随机抽样检查差异
            is_static = True
            first_frame = all_obs[0]
            for i in range(1, len(all_obs), max(1, len(all_obs)//10)):
                if not np.array_equal(first_frame, all_obs[i]):
                    is_static = False
                    break
            
            if is_static:
                print("\n[严重警告] 采集到的所有帧似乎完全相同！")
                print("可能原因：")
                print("1. 游戏窗口未激活或被遮挡")
                print("2. 摄像头/OBS采集源未更新")
                print("3. 采集帧率过高导致重复采样")
        
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n[错误] 采集失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    collect_data()
