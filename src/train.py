import os
import sys
import argparse
import shutil
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.interfaces.system.window as window_utils
from src.tasks.registration import task_registry
import src.tasks.sekiro  # 触发注册
import configs.config as config
from src.model.ppo_aux import AuxPPO
from src.envs.mdp.actions import ACTION_LABELS
from src.model.ppo_models import SekiroStableExtractor
from src.visualization.callbacks import SekiroCombinedCallback

def main():
    parser = argparse.ArgumentParser(description="Sekiro-RL PPO 训练入口 (Isaac Lab 风格)")
    parser.add_argument("--task", type=str, default="Sekiro-v0", help="要训练的任务 ID")
    parser.add_argument("--steps", type=int, default=100000, help="总训练时间步数")
    parser.add_argument("--save_freq", type=int, default=10000, help="模型保存频率 (steps)")
    parser.add_argument("--lr", type=float, default=3e-4, help="降低学习率以稳定高奖励环境 (原 3e-4)")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小 (减小以节省显存)")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO 采集步数 (减小以减少单次 Rollout 内存占用)")
    parser.add_argument("--aux_coef", type=float, default=0.1, help="特征余弦相似度辅助损失权重")
    parser.add_argument("--checkpoint", type=str, default='None', help="断点模型路径 (例如 models/ppo_checkpoints/sekiro_ppo_10000_steps.zip)")
    
    args = parser.parse_args()

    # 1. 创建环境
    print(f"正在启动任务: {args.task}")
    env, env_cfg = task_registry.make(args.task)
    
    # 2. 配置神经网络架构 (使用更稳定的特征提取器)
    policy_kwargs = dict(
        features_extractor_class=SekiroStableExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 256], vf=[512, 256]), # 减小 MLP 层宽度以提高泛化能力
        activation_fn=nn.ReLU
    )

    # 3. 初始化 PPO 模型与日志系统
    print(f"训练设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 统一日志路径并清理旧日志 (自动覆盖机制)
    tb_log_path = os.path.join(config.cfg.path.log_dir, "ppo_sekiro")
    if os.path.exists(tb_log_path):
        print(f"正在清理旧日志目录: {tb_log_path}")
        shutil.rmtree(tb_log_path)
    
    # 创建统一的 SB3 日志器
    new_logger = configure(tb_log_path, ["stdout", "csv", "tensorboard"])
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"正在从断点加载模型: {args.checkpoint}")
        model = AuxPPO.load(
            args.checkpoint, 
            env=env, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            custom_objects={
                "learning_rate": args.lr,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "aux_coef": args.aux_coef
            }
        )
    else:
        print("未指定有效断点，正在初始化新模型...")
        model = AuxPPO(
            "CnnPolicy", 
            env, 
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=2,
            ent_coef=0.01, # 增加熵系数，强迫模型探索，防止死锁在“左移”等单一动作
            clip_range=0.2, # 限制策略更新幅度
            max_grad_norm=0.5, # 显式开启梯度裁剪，防止第一轮更新干爆模型
            aux_coef=args.aux_coef,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    # 统一设置日志器 (覆盖 SB3 默认的子目录行为)
    model.set_logger(new_logger)

    # 4. 配置自动保存与可视化回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path="./models/ppo_checkpoints/",
        name_prefix="sekiro_ppo"
    )
    
    sekiro_callback = SekiroCombinedCallback(log_interval=1000)

    # 5. 开始训练
    print(f"训练开始。Tensorboard 日志目录: {config.cfg.path.log_dir}")
    try:
        model.learn(
            total_timesteps=args.steps, 
            callback=[checkpoint_callback, sekiro_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("训练被手动中断。")
    finally:
        # 保存最终模型
        model.save("models/sekiro_ppo_final")
        env.close()
        print("环境已关闭，最终模型已保存。")

if __name__ == "__main__":
    main()
