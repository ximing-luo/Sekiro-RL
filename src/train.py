import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.interfaces.system.window as window_utils
from src.tasks.registration import task_registry
import src.tasks.sekiro  # 触发注册
import configs.config as config
from src.visualization.tensorboard_utils import TensorboardHookManager

class SekiroCombinedCallback(BaseCallback):
    """
    全能回调类：
    1. 集成 Isaac Lab 风格打印（含奖励分量）。
    2. 集成卷积层特征图可视化。
    """
    def __init__(self, verbose=0, log_interval=1000):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.hook_manager = None
        self.iteration = 0
        # 奖励分量缓冲区
        self.reward_buffer = {}

    def _on_training_start(self):
        # 初始化特征图 Hook
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        writer = None
        for output_format in self.model.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                writer = output_format.writer
                break
        
        if writer is not None:
            self.hook_manager = TensorboardHookManager(self.model, writer, log_interval=self.log_interval)
            self.hook_manager.register_hooks()

    def _on_step(self) -> bool:
        # 1. 从 infos 提取奖励分量
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" in info:
                components = info["reward_components"]
                for name, val in components.items():
                    if name not in self.reward_buffer:
                        self.reward_buffer[name] = []
                    self.reward_buffer[name].append(val)
        return True

    def _on_rollout_end(self):
        self.iteration += 1
        print(f"\n[Step {self.num_timesteps}] 采集完成，正在开始反向传播训练...")
        
        # 1. 计算本轮奖励分量均值
        avg_rewards = {}
        for name, vals in self.reward_buffer.items():
            if len(vals) > 0:
                avg_rewards[name] = sum(vals) / len(vals)
        self.reward_buffer = {} # 清空缓冲区

        # 2. Isaac Lab 风格打印
        metrics = self.logger.name_to_value
        print("\n" + "="*50)
        print(f"Iteration {self.iteration: <3} | Total Steps: {self.num_timesteps: <8}")
        print("-" * 50)
        
        # 打印各分量 (按贡献排序)
        print("Reward Components (Avg per Step):")
        for name, val in sorted(avg_rewards.items(), key=lambda x: abs(x[1]), reverse=True):
            color = "\033[92m" if val >= 0 else "\033[91m" # 绿色正分，红色负分
            print(f"  - {name: <15}: {color}{val: .4f}\033[0m")
        
        print("-" * 50)
        # 打印总体指标
        if "rollout/ep_rew_mean" in metrics:
            print(f"Mean Episode Reward: {metrics['rollout/ep_rew_mean']:.2f}")
        if "train/loss" in metrics:
            print(f"Training Loss:       {metrics['train/loss']:.4f}")
        if "time/fps" in metrics:
            print(f"FPS:                 {int(metrics['time/fps'])}")
        print("="*50 + "\n")

    def _on_training_end(self):
        if self.hook_manager:
            self.hook_manager.remove_hooks()

class SekiroCNN(BaseFeaturesExtractor):
    """
    自定义卷积神经网络架构。
    SB3 会自动根据 observation_space 的维度调用此模块。
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] # 通常为 3 (RGB)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 计算卷积后的输出维度以连接全连接层
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def main():
    parser = argparse.ArgumentParser(description="Sekiro-RL PPO 训练入口 (Isaac Lab 风格)")
    parser.add_argument("--task", type=str, default="Sekiro-v0", help="要训练的任务 ID")
    parser.add_argument("--steps", type=int, default=1000000, help="总训练时间步数")
    parser.add_argument("--save_freq", type=int, default=5000, help="模型保存频率 (steps)")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--n_steps", type=int, default=1024, help="PPO 采集步数 (Rollout Steps)")
    
    args = parser.parse_args()

    # 1. 创建环境
    print(f"正在启动任务: {args.task}")
    env, env_cfg = task_registry.make(args.task)
    
    # 2. 配置神经网络架构 (CNN 特征提取器 + MLP 决策头)
    policy_kwargs = dict(
        features_extractor_class=SekiroCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]), # pi: 策略网络, vf: 价值网络
        activation_fn=nn.ReLU
    )

    # 3. 初始化 PPO 模型
    print(f"训练设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=4, # 显著提速：将训练轮数从 10 降到 4
        verbose=1,
        tensorboard_log=config.cfg.path.log_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

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
