import os
import sys
import shutil
import argparse

import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import configs.config as config
import src.envs.tasks.sekiro  # 触发注册
import src.interfaces.system.window as window_utils
from src.envs.mdp.actions import ACTION_LABELS
from src.envs.tasks.registration import task_registry
from src.model import SekiroMADSExtractor, SekiroMLPExtractor
from src.policies.sb3.policy import SekiroCustomPolicy
from src.policies.sb3.sekiro_ppo import SekiroPPO
from src.policies.visualization.callbacks import SekiroCombinedCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Sekiro-RL PPO 训练入口 (Isaac Lab 风格)")
    parser.add_argument("--task", type=str, default="Sekiro-v0", help="要训练的任务 ID")
    parser.add_argument("--steps", type=int, default=100000, help="总训练时间步数")
    parser.add_argument("--save_freq", type=int, default=10000, help="模型保存频率 (steps)")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--n_steps", type=int, default=64, help="PPO 采集步数")
    parser.add_argument("--checkpoint", type=str, default='None', help="断点模型路径")
    return parser.parse_args()

def setup_logger(log_dir):
    tb_log_path = os.path.join(log_dir, "ppo_sekiro")
    if os.path.exists(tb_log_path):
        print(f"正在清理旧日志目录: {tb_log_path}")
        shutil.rmtree(tb_log_path)
    return configure(tb_log_path, ["stdout", "csv", "tensorboard"])

def get_model(args, env, policy_kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"正在从断点加载模型: {args.checkpoint}")
        return SekiroPPO.load(args.checkpoint, env=env, device=device,
                            custom_objects={"learning_rate": args.lr, "n_steps": args.n_steps, "batch_size": args.batch_size})
    print("未指定有效断点，正在初始化新模型...")
    return SekiroPPO(SekiroCustomPolicy, env, policy_kwargs=policy_kwargs,
                    learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=2, ent_coef=0.01, clip_range=0.2, max_grad_norm=0.5,
                    verbose=1, device=device, vicreg_coef=1.0, inv_dyn_coef=0.1,
                    vf_coef=0.2, clip_range_vf=0.2)

def main():
    args = parse_args()
    print(f"正在启动任务: {args.task}")
    env, _ = task_registry.make(args.task)
    
    # 2. 配置神经网络架构 (MADS 双流提取器)
    policy_kwargs = dict(
        features_extractor_class=SekiroMADSExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 256], vf=[512, 256])
    )

    # 3. 初始化 PPO 模型与日志系统
    print(f"训练设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    model = get_model(args, env, policy_kwargs)
    model.set_logger(setup_logger(config.cfg.path.log_dir))

    # 4. 配置自动保存与可视化回调
    callbacks = [
        CheckpointCallback(save_freq=args.save_freq, save_path="./models/ppo_checkpoints/", name_prefix="sekiro_ppo"),
        SekiroCombinedCallback(log_interval=1000)
    ]

    # 5. 开始训练
    print(f"训练开始。Tensorboard 日志目录: {config.cfg.path.log_dir}")
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        print("训练被手动中断。")
    finally:
        model.save("models/sekiro_ppo_final")
        env.close()
        print("环境已关闭，最终模型已保存。")

if __name__ == "__main__":
    main()
