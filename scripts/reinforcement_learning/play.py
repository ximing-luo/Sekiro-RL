import os
import sys
import argparse
import torch
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# 1. 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tasks.registration import task_registry
from src.gamelab.app.app_launcher import AppLauncher
from src.gamelab.app.runners import SekiroRunner
from src.framework.sb3.ppo_aux import AuxPPO
import src.gamelab.interfaces.window_utils as window_utils
import configs.config as config
import cli_args

def main():
    # 1. 初始化启动器
    launcher = AppLauncher()
    
    # 2. 添加推理特定参数
    parser = argparse.ArgumentParser(description="Sekiro-RL 推理入口 (Isaac Lab 风格)", add_help=False)
    cli_args.add_sb3_args(parser)
    
    # 合并参数
    args, _ = parser.parse_known_args(namespace=launcher.args)
    args = cli_args.update_sb3_cfg(args)

    # 3. 配置组装 (对齐 IsaacLab：显式覆盖配置属性)
    env_cfg = task_registry.get_task_cfg(launcher.task_name)
    env_cfg.scene.num_envs = launcher.num_envs
    env_cfg.device = launcher.device
    env_cfg.headless = launcher.headless
    
    # 4. 创建环境 (传递已就绪的配置)
    env = task_registry.make(launcher.task_name, cfg=env_cfg)
    
    # 5. 加载 Agent (SB3 模型)
    model_path = args.checkpoint if args.checkpoint else config.cfg.path.model_path
    print(f"[INFO] 正在加载模型: {model_path}")
    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return

    # 加载归一化统计量 (如果存在)
    stats_path = model_path.replace(".zip", "_vec_normalize.pkl")
    if os.path.exists(stats_path):
        print(f"[INFO] 正在加载归一化统计量: {stats_path}")
        # 推理时需要包装环境以应用统计量
        temp_env = DummyVecEnv([lambda: env])
        temp_env = VecNormalize.load(stats_path, temp_env)
        temp_env.training = False
        temp_env.norm_reward = False
        env = temp_env

    # 加载模型
    agent = AuxPPO.load(model_path, device=launcher.device)
    
    # 5. 初始化执行器
    runner = SekiroRunner(env, agent, env_cfg)
    
    print(f"[INFO] 正在以推理模式运行任务: {launcher.task_name}")
    
    # 6. 执行推理循环
    steps_to_play = args.steps if args.steps else 10000
    try:
        # 重置环境
        env.reset()
        # 运行 runner
        runner.run(total_steps=steps_to_play)
    except KeyboardInterrupt:
        print("[WARN] 推理被手动中断。")
    finally:
        env.close()
        print("[INFO] 环境已关闭。")
        window_utils.move_window("Sekiro", "center")

if __name__ == "__main__":
    main()
