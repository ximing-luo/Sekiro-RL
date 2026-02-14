import time
import torch
import unittest
from collections import deque
from typing import Any, Optional, Dict, Protocol, runtime_checkable

from src.utils.import_utils import import_class

@runtime_checkable
class AgentProtocol(Protocol):
    """Agent 必须遵循的强硬契约。"""
    def predict(self, obs: Any, deterministic: bool = True) -> Any: ...

class Runner:
    """通用交互执行器基类。
    
    负责管理环境与 Agent 的生命周期，以及回调系统的调度。
    """

    def __init__(self, env: Any, agent: Optional[AgentProtocol] = None, env_cfg: Optional[Any] = None):
        self.env = env
        self.agent = agent
        self.env_cfg = env_cfg
        
        self.recent_rewards = deque(maxlen=100)
        self.callbacks = []
        self._init_callbacks()

    def _init_callbacks(self):
        """从环境配置中动态加载扩展/回调。"""
        # 物理化：假设 env_cfg.callbacks 必须存在（如果 env_cfg 存在）。
        # 如果不存在，说明配置结构定义错误，应当报错。
        if not self.env_cfg or not hasattr(self.env_cfg, "callbacks"):
             return

        for cb_cfg in self.env_cfg.callbacks:
            try:
                cb_class = import_class(cb_cfg["class"])
                # 物理化：参数必须是字典，且必须存在
                params = cb_cfg.get("params", {})
                cb_instance = cb_class(**params)
                # 注入引用
                cb_instance.model = self.agent
                cb_instance.env = self.env
                self.callbacks.append(cb_instance)
                print(f"[INFO] 已挂载回调: {cb_cfg['class']}")
            except Exception as e:
                # 这里的 try-except 是为了防止单个回调崩溃影响主流程，
                # 但根据“契约神圣性”，也许应该让它崩？
                # 暂且保留，打印错误。
                print(f"[ERROR] 加载回调 {cb_cfg.get('class')} 失败: {e}")

    def trigger_event(self, event_name: str, **kwargs):
        """分发生命周期事件给所有回调。"""
        # 内部约定：事件名对应私有方法（如 on_step -> _on_step）
        func_name = f"_{event_name}" if not event_name.startswith("_") else event_name
        
        for cb in self.callbacks:
            if event_name == "on_step":
                cb.locals = {"infos": [kwargs.get("info", {})]}
            
            # 契约：回调必须实现对应方法，否则 getattr 会抛出 AttributeError。
            # 我们不再 swallow 这个错误，让它成为“裁断”。
            if hasattr(cb, func_name):
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
            # 1. 采样观测
            obs = self.env.observation_manager.step()
            
            # 2. Agent 决策 (逻辑坍缩：在初始化阶段完成类型对齐)
            action = self._get_action(obs)

            # 3. 环境步进
            next_obs, reward, done, time_out, info = self.env.step(action)
            
            # 4. 数据统计与回调
            self.recent_rewards.append(reward.mean().item())
            self.trigger_event("on_step", info=info)
            
            # 5. 状态打印 (卫语句展平逻辑流)
            if time.time() - last_print_time < 1.0:
                continue
                
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0.0
            print(f"\033[92m[Step {step}] Avg Reward (100): {avg_reward:.4f} | Done: {done.any().item()}\033[0m")
            last_print_time = time.time()

    def _get_action(self, obs):
        """采样动作。不再使用 hasattr 试探，而是通过策略映射或强契约指代。"""
        if not self.agent:
            return torch.zeros((self.env.num_envs, self.env.action_dim), device=self.env.device)
        
        return self.agent.predict(obs, deterministic=True)[0]
