import os
import sys
import time
import argparse
import torch

# 1. 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gamelab.app.app_launcher import AppLauncher
from src.gamelab.app.runners import SekiroRunner
from src.framework.sb3.ppo_aux import AuxPPO
import src.gamelab.interfaces.window_utils as window_utils
import configs.config as config

def main():
    # 1. 初始化启动器
    launcher = AppLauncher()
    
    # 2. 添加推理特定参数
    parser = argparse.ArgumentParser(description="Sekiro-RL 推理入口 (Isaac Lab 风格)", add_help=False)
    play_group = parser.add_argument_group("推理特定参数")
    play_group.add_argument("--model_path", type=str, default=config.cfg.path.model_path, help="模型加载路径")
    play_group.add_argument("--steps", type=int, default=10000, help="运行总步数")
    
    # 合并参数
    args, _ = parser.parse_known_args(namespace=launcher.args)

    # 3. 创建环境
    env, env_cfg = launcher.create_env()
    
    # 4. 加载 Agent (SB3 模型)
    print(f"[INFO] 正在加载模型: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"[ERROR] 模型文件不存在: {args.model_path}")
        return

    # 加载模型
    agent = AuxPPO.load(args.model_path, device=launcher.device)
    
    # 5. 初始化执行器
    runner = SekiroRunner(env, agent, env_cfg)
    
    print(f"[INFO] 正在以推理模式运行任务: {launcher.task_name}")
    
    # 6. 执行推理循环
    try:
        # 重置环境
        env.reset()
        # 运行 runner
        runner.run(total_steps=args.steps)
    except KeyboardInterrupt:
        print("[WARN] 推理被手动中断。")
    finally:
        env.close()
        print("[INFO] 环境已关闭。")
        window_utils.move_window("Sekiro", "center")

if __name__ == "__main__":
    main()
