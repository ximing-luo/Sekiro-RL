# Copyright (c) 2024, Sekiro-RL Project.
# All rights reserved.

"""针对 SB3/PPO 的命令行参数处理，对齐 IsaacLab 风格。"""

import argparse
import random

def add_sb3_args(parser: argparse.ArgumentParser):
    """添加 Stable-Baselines3 相关的命令行参数。"""
    arg_group = parser.add_argument_group("sb3", description="SB3 智能体参数")
    
    # -- 实验参数
    arg_group.add_argument(
        "--experiment_name", type=str, default="ppo_sekiro", help="存储日志的实验文件夹名称。"
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="运行名称后缀。")
    
    # -- 加载参数
    arg_group.add_argument("--resume", action="store_true", default=False, help="是否从断点恢复训练。")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="要加载的断点文件路径。")
    
    # -- 超参数覆盖 (对齐 IsaacLab 的灵活性)
    # 默认值设为 None，表示不覆盖，使用 config.py 或任务配置中的值
    arg_group.add_argument("--steps", type=int, default=None, help="总训练时间步数。")
    arg_group.add_argument("--learning_rate", type=float, default=None, help="学习率。", dest="learning_rate")
    arg_group.add_argument("--batch_size", type=int, default=None, help="批大小。")
    arg_group.add_argument("--n_steps", type=int, default=None, help="PPO 采集步数。")
    arg_group.add_argument("--gamma", type=float, default=None, help="折扣因子。")
    arg_group.add_argument("--n_epochs", type=int, default=None, help="每轮更新的 Epoch 数量。")
    arg_group.add_argument("--target_kl", type=float, default=None, help="目标 KL 散度。")
    arg_group.add_argument("--ent_coef", type=float, default=None, help="熵系数。")
    arg_group.add_argument("--clip_range", type=float, default=None, help="PPO 裁剪范围。")
    arg_group.add_argument("--clip_range_vf", type=float, default=None, help="价值函数裁剪范围。")
    arg_group.add_argument("--max_grad_norm", type=float, default=None, help="最大梯度范数。")
    arg_group.add_argument("--aux_coef", type=float, default=None, help="特征相似度辅助损失权重。")
    arg_group.add_argument("--vf_coef", type=float, default=None, help="价值函数权重系数。")
    arg_group.add_argument("--save_freq", type=int, default=None, help="模型保存频率 (steps)。")

import configs.config as config

def update_sb3_cfg(args_cli: argparse.Namespace):
    """更新配置字典或 Namespace 以覆盖默认超参数。
    
    遵循分层优先级 (Isaac Lab 风格)：
    1. CLI Args (如果不是 None)
    2. Global Config (基准值，来自 config.py)
    """
    from dataclasses import asdict
    
    # 1. 提取全局配置中的所有参数作为基准
    # 我们将 train 模块的参数映射到 cli 命名空间
    train_defaults = asdict(config.cfg.train)
    
    # 2. 补充 CLI 特有但不在 TrainConfig 里的默认值
    defaults = {
        "run_name": "Sekiro",
        **train_defaults
    }

    # 3. 遍历并应用优先级逻辑：CLI 覆盖基准
    for key, default_val in defaults.items():
        if hasattr(args_cli, key) and getattr(args_cli, key) is None:
            setattr(args_cli, key, default_val)
    
    # 如果设置了 -1 则随机化种子
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
    
    return args_cli
