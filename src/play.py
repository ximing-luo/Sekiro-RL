import os
import sys
import time

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.interfaces.system.window as window_utils
from src.utils.train_utils import init_env_agent, wait_buffer, SekiroRunner
import configs.config as config

def play_agent(
    pos="offscreen",
    img_width=480,
    img_height=270,
    action_dim=None,
    total_interaction_steps=100000,
    model_path=config.MODEL_PATH,
    n_step_rewards: int = 20,
):
    """
    运行代理模型入口（推理模式）。
    """
    # 1. 初始化环境、代理与执行器
    env, agent = init_env_agent(pos, img_width, img_height, action_dim, model_path, n_step_rewards)
    runner = SekiroRunner(env, agent)
    
    print(f"正在以推理模式运行模型: {model_path}")
    
    # 2. 配置推理模式
    agent.eval()
    
    # 3. 准备运行环境
    env.pause_game(True)
    wait_buffer(env)
    
    # 4. 执行推理循环
    env.reset()
    runner.run_episode(0, 0, total_interaction_steps, is_train=False)
            
    print("运行结束。")

if __name__ == "__main__":
    play_agent()
    time.sleep(1.0)
    window_utils.move_window("Sekiro", "center")
