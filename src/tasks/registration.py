from dataclasses import dataclass
from typing import Type, Dict, Any, Tuple
import importlib

@dataclass
class TaskSpec:
    """任务规范类，保存环境类及其配置类。"""
    task_name: str
    env_class: Type
    env_cfg_class: Type

class TaskRegistry:
    """任务注册中心。"""
    def __init__(self):
        self._tasks: Dict[str, TaskSpec] = {}

    def register(self, task_name: str, env_class: Type, env_cfg_class: Type):
        """注册一个任务。"""
        if task_name in self._tasks:
            print(f"警告: 任务 {task_name} 已注册，将被覆盖。")
        self._tasks[task_name] = TaskSpec(task_name, env_class, env_cfg_class)
        print(f"已注册任务: {task_name}")

    def make(self, task_name: str, **kwargs) -> Tuple[Any, Any]:
        """根据任务名创建环境实例。"""
        if task_name not in self._tasks:
            raise ValueError(f"任务 {task_name} 未找到。请确保已注册。")
        
        spec = self._tasks[task_name]
        # 1. 实例化配置类
        cfg = spec.env_cfg_class()
        
        # 2. 应用命令行或额外的参数覆盖
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            elif hasattr(cfg.scene, key):
                setattr(cfg.scene, key, value)
        
        # 3. 实例化环境类
        env = spec.env_class(cfg=cfg)
        
        return env, cfg

    def get_task_names(self):
        return list(self._tasks.keys())

# 全局单例
task_registry = TaskRegistry()
