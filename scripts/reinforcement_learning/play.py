import os
import sys
import time
import argparse

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.gamelab.interfaces.window_utils as window_utils
from src.gamelab.app.runners import init_env_agent, wait_buffer, SekiroRunner
import configs.config as config

def main():
    parser = argparse.ArgumentParser(description="Sekiro-RL 推理入口 (Isaac Lab 风格)")
    parser.add_argument("--task", type=str, default="Sekiro-v0", help="要运行的任务 ID")
    parser.add_argument("--pos", type=str, default=config.cfg.scene.pos, help="窗口位置 (center/offscreen)")
    parser.add_argument("--steps", type=int, default=100000, help="总交互步数")
    parser.add_argument("--model_path", type=str, default=config.cfg.path.model_path, help="模型加载路径")
    
    args = parser.parse_args()

    # 1. 初始化环境、代理与执行器
    env, agent, env_cfg = init_env_agent(
        task_name=args.task,
        pos=args.pos,
        img_width=config.cfg.scene.img_width,
        img_height=config.cfg.scene.img_height,
        action_dim=None,
        model_path=args.model_path,
        n_step_rewards=config.cfg.rl.n_step_rewards
    )
    
    runner = SekiroRunner(env, agent, env_cfg)
    
    print(f"正在以推理模式运行任务: {args.task}, 模型: {args.model_path}")
    
    # 2. 配置推理模式
    agent.eval()
    
    # 3. 准备运行环境
    env.pause_game(True)
    wait_buffer(env)
    
    # 4. 执行推理循环
    env.reset()
    runner.run_episode(0, 0, args.steps, is_train=False)
            
    print("运行结束。")
    time.sleep(1.0)
    window_utils.move_window("Sekiro", "center")

if __name__ == "__main__":
    main()
