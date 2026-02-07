import time
import torch
import unittest
from collections import deque
from typing import Any, Optional, Dict

from src.utils.import_utils import import_class

def run_tests(verbosity: int = 2, **kwargs):
    """对标 Isaac Lab 的测试运行器封装。"""
    unittest.main(verbosity=verbosity, exit=True, **kwargs)

class Runner:
    """通用交互执行器基类。
    
    负责管理环境与 Agent 的生命周期，以及回调系统的调度。
    """

    def __init__(self, env: Any, agent: Optional[Any] = None, env_cfg: Optional[Any] = None):
        self.env = env
        self.agent = agent
        self.env_cfg = env_cfg
        
        self.recent_rewards = deque(maxlen=100)
        self.callbacks = []
        self._init_callbacks()

    def _init_callbacks(self):
        """从环境配置中动态加载扩展/回调。"""
        if self.env_cfg and hasattr(self.env_cfg, "callbacks"):
            for cb_cfg in self.env_cfg.callbacks:
                try:
                    cb_class = import_class(cb_cfg["class"])
                    cb_instance = cb_class(**cb_cfg.get("params", {}))
                    # 注入引用
                    cb_instance.model = self.agent
                    cb_instance.env = self.env
                    self.callbacks.append(cb_instance)
                    print(f"[INFO] 已挂载回调: {cb_cfg['class']}")
                except Exception as e:
                    print(f"[ERROR] 加载回调 {cb_cfg.get('class')} 失败: {e}")

    def trigger_event(self, event_name: str, **kwargs):
        """分发生命周期事件给所有回调。"""
        for cb in self.callbacks:
            # 兼容不同回调库的命名习惯 (如 SB3 的 _on_step)
            func_name = f"_{event_name}" if not event_name.startswith("_") else event_name
            if hasattr(cb, func_name):
                # 注入上下文
                if event_name == "on_step":
                    cb.locals = {"infos": [kwargs.get("info", {})]}
                getattr(cb, func_name)()

    def run(self, total_steps: int):
        """执行循环的主入口。"""
        raise NotImplementedError("Subclasses must implement run()")

class SekiroRunner(Runner):
    """Sekiro 任务专用执行器。
    
    实现了具体的拼刀节奏控制、暂停处理和数据流转逻辑。
    """

    def run(self, total_steps: int):
        print(f"[INFO] 开始 Sekiro 运行循环 (目标步数: {total_steps})...")
        self.trigger_event("on_training_start")
        
        last_print_time = time.time()
        
        for step in range(total_steps):
            # 1. 获取观测 (对标 Isaac Lab 的 obs 数据流)
            obs = self.env.observation_manager.compute_observations()
            
            # 2. Agent 决策 (支持 SB3 模型和自定义 Agent)
            if self.agent:
                if hasattr(self.agent, "predict"):
                    # SB3 风格
                    action, _ = self.agent.predict(obs, deterministic=True)
                elif hasattr(self.agent, "choose_action"):
                    # 自定义风格
                    action = self.agent.choose_action(obs)
                else:
                    raise ValueError("Agent 必须实现 predict() 或 choose_action() 方法")
            else:
                # 默认空动作或随机动作
                action = torch.zeros((self.env.num_envs, self.env.action_dim), device=self.env.device)

            # 3. 环境步进
            next_obs, reward, done, time_out, info = self.env.step(action)
            
            # 4. 数据统计与回调
            self.recent_rewards.append(reward.mean().item())
            self.trigger_event("on_step", info=info)
            
            # 5. 状态打印
            if time.time() - last_print_time > 1.0:
                avg_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0.0
                print(f"\033[92m[Step {step}] Avg Reward (100): {avg_reward:.4f} | Done: {done.any().item()}\033[0m")
                last_print_time = time.time()

            # 6. 处理终止
            if done.any() and self.env.num_envs == 1:
                # 单环境模式下，如果任务结束则重置（多环境由 ManagerBasedEnv 自动处理）
                self.env.reset()

        print("[INFO] Sekiro 运行结束。")
