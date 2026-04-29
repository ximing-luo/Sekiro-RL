#!/usr/bin/env python3
"""
CartPole-v1 可视化推理脚本
加载训练好的模型权重，渲染运行效果。
"""
import sys
import os
import argparse
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from src.framework.ashina.algorithm.modelfree.dqn import DQNAlgorithm
from src.framework.ashina.data.batch import Batch


def play(args):
    env = gym.make("CartPole-v1")
    algo = DQNAlgorithm(action_dim=env.action_space.n, input_dim=4, lr=0)

    if args.model:
        algo.policy.model.load_state_dict(
            torch.load(args.model, map_location="cpu", weights_only=True)
        )
        algo.sync_target()
        print(f"已加载模型: {args.model}")
    else:
        print("未指定模型，使用随机初始化权重（仅用于测试渲染）")

    algo.policy.eval()

    env = gym.make("CartPole-v1", render_mode="human")
    episode_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = algo(Batch(obs=np.array([obs]))).act.item()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if done or truncated:
                break

        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1:3d} | steps: {steps:4d} | reward: {total_reward:.1f}")

    env.close()
    print(f"\n平均奖励: {np.mean(episode_rewards):.1f}  "
          f"最高: {np.max(episode_rewards):.1f}  "
          f"最低: {np.min(episode_rewards):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default=None, help="模型权重路径 (.pth)")
    parser.add_argument("--episodes", type=int, default=5,   help="运行局数")
    args = parser.parse_args()
    play(args)
