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

def collect_data(task_id="Sekiro-v0", num_frames=512, fps=30):
    """
    启动只狼环境，通过随机动作采集真实观测数据并保存。
    """
    print(f"\n" + "="*50)
    print(f"正在启动环境以采集数据: {task_id} | 目标速度: {fps} FPS")
    print("="*50)
    
    try:
        # 通过 kwargs 覆盖配置中的 capture_fps
        env, env_cfg = task_registry.make(task_id, capture_fps=fps)
        
        obs, info = env.reset()
        all_obs = []
        
        print(f"\n准备就绪！开始采集 {num_frames} 帧数据...")
        start_time = time.time()
        interval = 1.0 / fps
        
        for i in range(num_frames):
            loop_start = time.time()
            
            # 随机动作以获得多样化画面
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # obs 是 (12, 135, 240) 的 uint8 数组
            all_obs.append(obs)
            
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
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n[错误] 采集失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    collect_data()
