import os
import time
import torch
import threading
import numpy as np
from collections import deque
from src.tasks.registration import task_registry
import src.tasks.sekiro # 确保 Sekiro 任务已注册
from src.policies.dqn_agent import DQNAgent
from src.interfaces.system.input import key_check
import configs.config as config

def init_env_agent(task_name, pos, img_width, img_height, action_dim, model_path, n_step_rewards):
    """职责：通过注册表初始化环境与代理模型。"""
    # 1. 从注册表创建环境
    env, env_cfg = task_registry.make(
        task_name, 
        pos=pos, 
        observation_w=img_width, 
        observation_h=img_height,
        n_step_rewards=n_step_rewards
    )
    
    # 2. 确定动作空间维度
    ad = action_dim if action_dim is not None else env.action_dim
    
    # 3. 初始化 Agent
    agent = DQNAgent(img_width, img_height, ad, buffer=env.replay_buffer, model_file=model_path, n_step_rewards=n_step_rewards)
    
    if os.path.isfile(model_path):
        try:
            agent.load()
            print(f"已从现有模型恢复：{model_path}")
        except Exception as e:
            print(f"加载已有模型失败（忽略继续训练）：{e}")
    return env, agent

def wait_buffer(env):
    """职责：等待环境的视频缓冲区加载满，确保初始状态稳定。"""
    print("等待缓冲区填充初始帧...")
    while len(env.replay_buffer) < env.replay_buffer.frame_history_len:
        # 手动执行一些空动作来填充缓冲区
        env.step(0) # 新接口下这里会返回 5 个值，但我们只需触发 step
        time.sleep(0.05)
    time.sleep(0.2)

def maybe_print(step, recent_rewards, last_print_time):
    """职责：定期打印训练/运行状态。"""
    if time.time() - last_print_time > 1.0:
        avg = (sum(recent_rewards) / len(recent_rewards)) if len(recent_rewards) > 0 else 0.0
        print(f"\033[91mstep={step} avg_reward_10={avg:.3f}\033[0m")
        return time.time()
    return last_print_time

class SekiroRunner:
    """
    Sekiro 任务执行器：封装训练与推理的核心循环逻辑。
    遵循 Isaac Lab 的解耦理念，将环境交互、动作选择、日志记录等逻辑统一管理。
    """
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.recent_rewards = deque(maxlen=10)
        self.last_print_time = time.time()
        self.last_save_steps = 0

    def run_episode(self, episode_idx, global_step, total_steps, is_train=True):
        """运行单个训练/推理回合。"""
        self.env.reset()
        self.last_print_time = time.time()
        
        for _ in range(total_steps):
            loop_start = time.time()
            
            # 1. 计算探索率并更新步数
            epsilon = 0.0
            if is_train:
                # 使用 config 中的探索参数
                eps_start = config.cfg.epsilon.eps_start
                eps_end = config.cfg.epsilon.eps_end
                eps_decay = config.cfg.epsilon.eps_decay
                epsilon = eps_end + (eps_start - eps_end) * max(0, (eps_decay - global_step) / eps_decay)
            
            global_step += 1
            self.agent.current_step = global_step
            
            # 2. 获取当前观测并选择动作
            k = self.env.replay_buffer.frame_history_len
            state = self.env.replay_buffer.get_latest_observation(k)
            self.env.render_debug(state)
            
            action = self.agent.act(state, epsilon=epsilon)
            
            # 3. 与环境交互 (使用标准 Gymnasium 接口)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.recent_rewards.append(float(reward))

            # 4. 训练优化 (仅训练模式)
            if is_train:
                self._maybe_optimize(global_step)

            # 5. 日志记录与状态打印
            dt = time.time() - loop_start
            fps = (1.0 / dt) if dt > 1e-6 else 0.0
            
            if is_train:
                self._log_metrics(global_step, episode_idx, action, reward, fps, epsilon)
            else:
                if global_step % 10 == 0:
                    print(f"FPS={fps:.2f} Reward={reward:.3f}")

            self.last_print_time = maybe_print(global_step, self.recent_rewards, self.last_print_time)

            # 6. 模型保存 (仅训练模式)
            if is_train:
                self.last_save_steps = self._maybe_save(self.last_save_steps)

            # 7. 游戏控制与暂停逻辑
            self.env.pause_game(False)
            if terminated or truncated or "P" in key_check():
                self.env.over = True
                break
            
            # 8. 帧率控制
            target_fps = config.cfg.scene.target_fps
            target_dt = 1.0 / target_fps
            actual_dt = time.time() - loop_start
            if actual_dt < target_dt:
                time.sleep(target_dt - actual_dt)
                
        if is_train:
            self._maybe_save(self.last_save_steps, force=True)
            
        return global_step

    def _log_metrics(self, step, episode, action, reward, fps, epsilon):
        """记录训练指标。"""
        self.env.log_manager.log_step(self.agent, self.env, step, episode, action, reward, self.recent_rewards, fps, epsilon)

    def _maybe_optimize(self, step):
        """触发异步模型学习。"""
        freq = config.cfg.train.optimize_every_steps
        if step % int(freq) == 0:
            threading.Thread(target=self.agent.learn, daemon=True).start()

    def _maybe_save(self, last_save_steps, force=False):
        """定期保存模型。"""
        if self.agent.optimize_count - last_save_steps >= 100 or force:
            self.agent.save()
            print(f"模型已保存：{self.agent.model_file}")
            return self.agent.optimize_count
        return last_save_steps
