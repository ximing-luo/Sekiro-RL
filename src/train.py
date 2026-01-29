import os
import sys
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.interfaces.system.window as window_utils
from src.visualization.logger import load_last_training_stats
from src.visualization.tensorboard_utils import register_tensorboard_hooks
from src.utils.train_utils import init_env_agent, wait_buffer, SekiroRunner
import configs.config as config

def main():
    parser = argparse.ArgumentParser(description="Sekiro-RL 训练入口 (Isaac Lab 风格)")
    parser.add_argument("--task", type=str, default="Sekiro-v0", help="要训练的任务 ID")
    parser.add_argument("--pos", type=str, default=config.cfg.scene.pos, help="窗口位置 (center/offscreen)")
    parser.add_argument("--lr", type=float, default=config.cfg.train.lr, help="学习率")
    parser.add_argument("--steps", type=int, default=10000, help="总交互步数")
    parser.add_argument("--model_path", type=str, default=config.cfg.path.model_path, help="模型保存路径")
    
    args = parser.parse_args()

    # 1. 初始化环境、代理与执行器
    # 注意：init_env_agent 现在通过注册表创建环境
    env, agent = init_env_agent(
        task_name=args.task,
        pos=args.pos,
        img_width=config.cfg.scene.img_width,
        img_height=config.cfg.scene.img_height,
        action_dim=None,
        model_path=args.model_path,
        n_step_rewards=config.cfg.rl.n_step_rewards
    )
    
    agent.train()
    runner = SekiroRunner(env, agent)
    
    # 2. 确保目录存在并配置可视化
    os.makedirs(config.cfg.path.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    writer = SummaryWriter(log_dir=config.cfg.path.log_dir)
    register_tensorboard_hooks(agent, writer, log_interval=config.cfg.path.tb_log_interval)
    
    # 3. 加载历史训练计数
    last_step, last_episode = load_last_training_stats(config.cfg.path.log_dir)
    global_step = int(last_step)
    
    # 4. 准备训练环境
    env.pause_game(True)
    wait_buffer(env)
    
    # 5. 主训练循环
    print(f"开始任务: {args.task}，从 Episode {last_episode + 1} 继续训练...")
    for episode in range(int(last_episode) + 1, int(last_episode) + 1 + 10):
        global_step = runner.run_episode(episode, global_step, args.steps, is_train=True)
        
        print(f"Episode {episode} 训练结束，累计步数: {global_step}")
        # 如果是因为 P 键手动停止或环境结束
        if getattr(env, 'over', False):
            break
            
    print("所有训练任务已完成。")
    time.sleep(1.0)
    window_utils.move_window("Sekiro", "center")

if __name__ == "__main__":
    main()
