import os
import sys
import time
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

def train_agent(
    pos="offscreen",
    img_width=480,
    img_height=270,
    action_dim=None,
    total_interaction_steps=10000,
    model_path=config.MODEL_PATH,
    n_step_rewards: int = 20,
):
    """
    训练代理模型入口。
    """
    # 1. 初始化环境、代理与执行器
    env, agent = init_env_agent(pos, img_width, img_height, action_dim, model_path, n_step_rewards)
    agent.train()
    runner = SekiroRunner(env, agent)
    
    # 2. 确保目录存在并配置可视化
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    register_tensorboard_hooks(agent, writer)
    
    # 3. 加载历史训练计数
    last_step, last_episode = load_last_training_stats(config.LOG_DIR)
    global_step = int(last_step)
    
    # 4. 准备训练环境
    env.pause_game(True)
    wait_buffer(env)
    
    # 5. 主训练循环（分回合运行）
    print(f"开始从 Episode {last_episode + 1} 继续训练...")
    for episode in range(int(last_episode) + 1, int(last_episode) + 1 + 10):
        global_step = runner.run_episode(episode, global_step, total_interaction_steps, is_train=True)
        
        print(f"Episode {episode} 训练结束，累计步数: {global_step}")
        if env.over:
            break
            
    print("所有训练任务已完成。")

if __name__ == "__main__":
    train_agent()
    time.sleep(1.0)
    window_utils.move_window("Sekiro", "center")
