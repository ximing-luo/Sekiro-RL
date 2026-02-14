# Copyright (c) 2024, Sekiro-RL Project.
# All rights reserved.

"""使用 SB3/PPO 训练 Sekiro 智能体的入口，对齐 IsaacLab 风格。"""

import os
import sys
import argparse
import torch
from datetime import datetime
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# 1. 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 导入架构组件
from src.tasks.registration import task_registry
from src.gamelab.app.app_launcher import AppLauncher
from src.framework.sb3.runner import SB3OnPolicyRunner
from src.framework.sb3.adapter import SB3VecEnvAdapter
from src.utils.io import dump_yaml, get_git_hash
import cli_args

def main():
    # 阶段一：初始化性能优化 (对齐 IsaacLab)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # 1. 初始化启动器
    launcher = AppLauncher()
    
    # 2. 解析参数
    parser = argparse.ArgumentParser(description="Sekiro-RL PPO 训练入口 (Isaac Lab 风格)", add_help=False)
    cli_args.add_sb3_args(parser)
    args, _ = parser.parse_known_args(namespace=launcher.args)
    args = cli_args.update_sb3_cfg(args)

    # 3. 配置组装 (对齐 IsaacLab：显式覆盖配置属性)
    env_cfg = task_registry.get_task_cfg(launcher.task_name)
    env_cfg.scene.num_envs = launcher.num_envs
    env_cfg.device = launcher.device
    env_cfg.headless = launcher.headless
    
    # 4. 创建环境 (传递已就绪的配置)
    env = task_registry.make(launcher.task_name, cfg=env_cfg)
    
    # 阶段二：环境包装 (对齐 IsaacLab 适配器模式)
    # 强硬契约：将原生 Tensor 环境包装为 SB3 标准 VecEnv，消除维度与类型歧义
    env = SB3VecEnvAdapter(env, device=launcher.device)
    
    # 启用奖励归一化 (可选)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.)
    
    # 阶段三：日志与实验管理
    log_root_path = os.path.join("logs", "sb3", args.experiment_name)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        log_dir += f"_{args.run_name}"
    log_dir = os.path.abspath(os.path.join(log_root_path, log_dir))
    
    # 备份配置与 Git 状态
    config_dir = os.path.join(log_dir, "params")
    dump_yaml(os.path.join(config_dir, "env.yaml"), env_cfg.__dict__ if hasattr(env_cfg, "__dict__") else env_cfg)
    dump_yaml(os.path.join(config_dir, "agent.yaml"), vars(args))
    with open(os.path.join(log_dir, "git_status.txt"), "w") as f:
        f.write(f"Git Hash: {get_git_hash()}\n")

    # 阶段四：启动训练
    runner = SB3OnPolicyRunner(env, args, log_dir=log_dir, device=launcher.device)
    runner.learn()

    # 训练结束，关闭环境
    env.close()

if __name__ == "__main__":
    main()
