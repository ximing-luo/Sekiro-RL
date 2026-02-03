import argparse
import os
import sys
import time

import torch

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import configs.config as config
import src.envs.tasks.sekiro  # 触发注册
import src.interfaces.system.window as window_utils
from src.envs.tasks.registration import task_registry
from src.policies.sb3.sekiro_ppo import SekiroPPO

def parse_args():
    parser = argparse.ArgumentParser(description="Sekiro-RL 推理入口 (Isaac Lab 风格)")
    parser.add_argument("--task", type=str, default="Sekiro-v0", help="要运行的任务 ID")
    parser.add_argument("--checkpoint", type=str, default="models/sekiro_ppo_final.zip", help="模型加载路径")
    parser.add_argument("--steps", type=int, default=100000, help="总推理步数")
    parser.add_argument("--deterministic", type=bool, default=True, help="是否使用确定性动作")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 创建环境
    print(f"正在启动环境: {args.task}")
    env, _ = task_registry.make(args.task)
    
    # 2. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.checkpoint):
        # 尝试去掉 .zip 后缀或补上
        if not args.checkpoint.endswith(".zip") and os.path.exists(args.checkpoint + ".zip"):
            args.checkpoint += ".zip"
        else:
            print(f"错误: 找不到模型文件 {args.checkpoint}")
            return

    print(f"正在加载模型: {args.checkpoint} (设备: {device})")
    model = SekiroPPO.load(args.checkpoint, env=env, device=device)
    
    # 3. 运行推理循环
    print(f"推理开始。按 'P' 键或 'Ctrl+C' 退出。")
    env.pause_game(True)
    obs, info = env.reset()
    
    try:
        for i in range(args.steps):
            # 获取 12 通道堆叠帧 (3通道 * 4帧)
            stacked_obs = env.observation_manager.get_latest_stacked_frames()
            
            # 模型预测
            action, _ = model.predict(stacked_obs, deterministic=args.deterministic)
            
            # 环境步进
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("回合结束，重置中...")
                env.pause_game(True)
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("推理被手动中断。")
    finally:
        print("正在清理环境...")
        env.close()
        window_utils.move_window("Sekiro", "center")
        print("运行结束。")

if __name__ == "__main__":
    main()
