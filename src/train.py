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
from src.envs.mdp.actions import ACTION_LABELS
from stable_baselines3.common.policies import ActorCriticPolicy
from src.policies.sekiro_ppo import SekiroPPO

class GroupLRWrapper(dict):
    """包装参数组字典，拦截 'lr' 设置并应用缩放比例"""
    def __init__(self, group, multiplier):
        super().__init__(group)
        self.multiplier = multiplier
    
    def __setitem__(self, key, value):
        if key == 'lr':
            # 当 SB3 尝试设置 lr 时，乘以预设的比例
            super().__setitem__(key, value * self.multiplier)
        else:
            super().__setitem__(key, value)

class SekiroCustomPolicy(ActorCriticPolicy):
    """自定义策略类，支持为特征提取器设置独立学习率并使用带归一化的 MLP 头部"""
    
    def _build_mlp_extractor(self) -> None:
        """重写以使用带 RMSNorm 的 SekiroMLPExtractor"""
        self.mlp_extractor = SekiroMLPExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _make_optimizer(self) -> torch.optim.Optimizer:
        # 定义参数组及其缩放比例
        # 让特征提取器（眼睛）的学习率始终是总学习率的 0.1 倍
        param_groups = [
            {"params": self.features_extractor.parameters(), "lr_multiplier": 0.1},
            {"params": self.mlp_extractor.parameters(), "lr_multiplier": 1.0},
            {"params": self.action_net.parameters(), "lr_multiplier": 1.0},
            {"params": self.value_net.parameters(), "lr_multiplier": 1.0},
        ]
        
        # 使用基础学习率初始化优化器
        base_lr = self.lr_schedule(1)
        optimizer = torch.optim.Adam(param_groups, lr=base_lr, **self.optimizer_kwargs)
        
        # 包装 param_groups 以拦截后续的自动更新
        new_groups = []
        for group in optimizer.param_groups:
            multiplier = group.pop("lr_multiplier", 1.0)
            new_groups.append(GroupLRWrapper(group, multiplier))
        
        optimizer.param_groups = new_groups
        return optimizer
from src.model.ppo_models import SekiroMADSExtractor
from src.model.mlp_heads import SekiroMLPExtractor
from src.visualization.callbacks import SekiroCombinedCallback

def main():
    parser = argparse.ArgumentParser(description="Sekiro-RL PPO 训练入口 (Isaac Lab 风格)")
    parser.add_argument("--task", type=str, default="Sekiro-v0", help="要训练的任务 ID")
    parser.add_argument("--steps", type=int, default=100000, help="总训练时间步数")
    parser.add_argument("--save_freq", type=int, default=10000, help="模型保存频率 (steps)")
    parser.add_argument("--lr", type=float, default=3e-4, help="降低学习率以稳定高奖励环境 (原 3e-4)")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小 (减小以节省显存)")
    parser.add_argument("--n_steps", type=int, default=64, help="PPO 采集步数 (减小以减少单次 Rollout 内存占用)")
    parser.add_argument("--checkpoint", type=str, default='None', help="断点模型路径 (例如 models/ppo_checkpoints/sekiro_ppo_10000_steps.zip)")
    
    args = parser.parse_args()

    # 1. 创建环境
    print(f"正在启动任务: {args.task}")
    env, env_cfg = task_registry.make(args.task)
    
    # 2. 配置神经网络架构 (MADS 双流提取器)
    policy_kwargs = dict(
        features_extractor_class=SekiroMADSExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 256], vf=[512, 256])
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
        model = SekiroPPO.load(
            args.checkpoint, 
            env=env, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            custom_objects={
                "learning_rate": args.lr,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size
            }
        )
    else:
        print("未指定有效断点，正在初始化新模型...")
        model = SekiroPPO(
            SekiroCustomPolicy, 
            env, 
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=2,
            ent_coef=0.01,
            clip_range=0.2,
            max_grad_norm=0.5,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            vicreg_coef=1.0,
            inv_dyn_coef=0.1,
            vf_coef=0.2, # 降低价值损失权重，保护共享特征提取器 (原 0.5)
            clip_range_vf=0.2 # 启用价值裁剪，防止数值爆炸
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
