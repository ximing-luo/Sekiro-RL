#!/usr/bin/env python3
"""
CartPole-v1 DQN 例程（迭代式并行训练）
演示 DQNAgent（high）和手动组装（low）两种用法。
"""
import sys
import os
import argparse
import datetime
import torch
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from src.framework.ashina.highlevel.dqn_agent import DQNAgent, DQNConfig
from src.framework.ashina.algorithm.modelfree.dqn import DQNAlgorithm
from src.framework.ashina.algorithm.net.discrete import QNetwork
from src.framework.ashina.data.buffer import ReplayBuffer
from src.framework.ashina.data.collector import Collector
from src.framework.ashina.trainer.offpolicy import OffPolicyTrainer
from src.framework.ashina.env.gymnasium_wrapper import GymnasiumWrapper


def make_log_dir(base: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"cartpole_dqn_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


def run_highlevel(args):
    """用法一：DQNAgent 高层接口"""
    env_fns = [lambda: gym.make("CartPole-v1") for _ in range(args.num_envs)]
    vec_env = GymnasiumWrapper(gym.vector.SyncVectorEnv(env_fns))

    agent = DQNAgent(
        vec_env=vec_env,
        action_dim=2,
        buffer=ReplayBuffer(size=args.buffer_size),
        params=DQNConfig(
            lr=args.lr,
            gamma=args.gamma,
            target_update_freq=args.target_update_freq,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            batch_size=args.batch_size,
        ),
    )

    print(f"Warmup: filling buffer with {args.warmup} steps...")
    agent.collector.collect_random_steps(args.warmup)

    for it in range(args.num_iterations):
        result = agent.learn(steps_per_iter=args.steps_per_iter, learn_epochs=args.learn_epochs)

        if it % 10 == 0:
            rew = result.get("rew", 0.0)
            bpe = result.get("batches_per_epoch", 0)
            epochs = result.get("epochs", 0)
            total_updates = bpe * epochs
            print(f"Iter {it:4d} | rew: {rew:.2f} | updates: {total_updates} ({bpe}×{epochs})")

    vec_env.close()

    log_dir = make_log_dir(args.log_dir)
    save_path = os.path.join(log_dir, "model.pth")
    agent.save(save_path)
    print(f"模型已保存至: {save_path}")


def run_lowlevel(args):
    """用法二：手动组装 Collector / Trainer"""
    env_fns = [lambda: gym.make("CartPole-v1") for _ in range(args.num_envs)]
    vec_env = GymnasiumWrapper(gym.vector.SyncVectorEnv(env_fns))
    obs_dim = 4
    action_dim = 2

    model = QNetwork(input_dim=obs_dim, action_dim=action_dim, hidden_dims=[128, 128])
    algo = DQNAlgorithm(
        action_dim=action_dim,
        model=model,
        lr=args.lr,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
    )
    buffer = ReplayBuffer(size=args.buffer_size)
    collector = Collector(algo, vec_env, buffer)
    trainer = OffPolicyTrainer(algo, collector, batch_size=args.batch_size)

    print(f"Warmup: filling buffer with {args.warmup} steps...")
    trainer.train_collector.collect_random_steps(args.warmup)

    for it in range(args.num_iterations):
        result = trainer.train_iteration(steps_per_iter=args.steps_per_iter, learn_epochs=args.learn_epochs)

        if it % 10 == 0:
            rew = result.get("rew", 0.0)
            bpe = result.get("batches_per_epoch", 0)
            epochs = result.get("epochs", 0)
            total_updates = bpe * epochs
            print(f"Iter {it:4d} | rew: {rew:.2f} | updates: {total_updates} ({bpe}×{epochs})")

    vec_env.close()

    log_dir = make_log_dir(args.log_dir)
    save_path = os.path.join(log_dir, "model.pth")
    torch.save(algo.policy.model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     choices=["high", "low"], default="high",
                        help="high=DQNAgent高层接口  low=手动组装低层接口")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="无头模式")
    parser.add_argument("--log-dir",             type=str,   default="logs/cartpole")
    parser.add_argument("--num-iterations",      type=int,   default=100,
                        help="总迭代次数")
    parser.add_argument("--steps-per-iter",      type=int,   default=24,
                        help="每迭代每个环境跑的步数")
    parser.add_argument("--learn-epochs",        type=int,   default=4,
                        help="每迭代学习轮次（采集的数据反复学几轮）")
    parser.add_argument("--warmup",              type=int,   default=5000,
                        help="预热步数")
    parser.add_argument("--buffer-size",         type=int,   default=100000)
    parser.add_argument("--batch-size",          type=int,   default=2048,
                        help="每次梯度更新的样本量")
    parser.add_argument("--lr",                  type=float, default=1e-3)
    parser.add_argument("--gamma",               type=float, default=0.99)
    parser.add_argument("--target-update-freq",  type=int,   default=500)
    parser.add_argument("--num-envs",            type=int,   default=1024,
                        help="并行环境数")
    args = parser.parse_args()

    if args.mode == "high":
        run_highlevel(args)
    else:
        run_lowlevel(args)
