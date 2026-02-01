import os
import sys
import time
import argparse
import shutil
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.interfaces.system.window as window_utils
from src.tasks.registration import task_registry
import src.tasks.sekiro  # 触发注册
import configs.config as config
from src.model.ppo_models import SekiroStableExtractor
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
        # 累计死亡统计 (训练开始至今)
        self.cum_player_deaths = 0
        self.cum_enemy_deaths = 0

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
            
            # 2. 统计死亡事件
            if "events" in info:
                events = info["events"]
                if 0 in events: 
                    self.cum_player_deaths += 1
                if 1 in events: 
                    self.cum_enemy_deaths += 1
        return True

    def _on_rollout_end(self):
        self.iteration += 1
        print(f"\n[Step {self.num_timesteps}] 采集完成，正在开始反向传播训练...")
        
        # 1. 计算本轮奖励分量均值
        avg_rewards = {}
        rollout_mean_reward = 0.0
        for name, vals in self.reward_buffer.items():
            if len(vals) > 0:
                avg_val = sum(vals) / len(vals)
                avg_rewards[name] = avg_val
                rollout_mean_reward += avg_val
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
        if "train/loss" in metrics:
            print(f"Training Loss:       {metrics['train/loss']:.4f}")
        if "time/fps" in metrics:
            print(f"FPS:                 {int(metrics['time/fps'])}")
        
        # 打印累计汇总 (本轮)
        print("-" * 50)
        print(f"Rollout Reward (Mean): {rollout_mean_reward:.4f}")
        if "rollout/ep_rew_mean" in metrics:
            print(f"Episode Reward (Mean): {metrics['rollout/ep_rew_mean']:.2f}")
        
        print(f"Cumulative Deaths -> Player: {self.cum_player_deaths}, Enemy: {self.cum_enemy_deaths}")
        
        print("="*50 + "\n")

    def _on_training_end(self):
        if self.hook_manager:
            self.hook_manager.remove_hooks()

def main():
    parser = argparse.ArgumentParser(description="Sekiro-RL PPO 训练入口 (Isaac Lab 风格)")
    parser.add_argument("--task", type=str, default="Sekiro-v0", help="要训练的任务 ID")
    parser.add_argument("--steps", type=int, default=100000, help="总训练时间步数")
    parser.add_argument("--save_freq", type=int, default=10000, help="模型保存频率 (steps)")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小 (减小以节省显存)")
    parser.add_argument("--n_steps", type=int, default=512, help="PPO 采集步数 (减小以减少单次 Rollout 内存占用)")
    parser.add_argument("--checkpoint", type=str, default='None', help="断点模型路径 (例如 models/ppo_checkpoints/sekiro_ppo_10000_steps.zip)")
    
    args = parser.parse_args()

    # 1. 创建环境
    print(f"正在启动任务: {args.task}")
    env, env_cfg = task_registry.make(args.task)
    
    # 2. 配置神经网络架构 (使用更稳定的特征提取器)
    policy_kwargs = dict(
        features_extractor_class=SekiroStableExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 512, 256], vf=[256, 512, 256]), # pi: 策略网络, vf: 价值网络
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
        model = PPO.load(
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
        model = PPO(
            "CnnPolicy", 
            env, 
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=2, # 压榨性能：增加训练轮数，确保反向传播停顿接近 2s 且学习更充分
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
