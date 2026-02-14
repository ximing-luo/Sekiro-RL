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
        self._tasks[task_name] = TaskSpec(task_name, env_class, env_cfg_class)
        print(f"已注册任务: {task_name}")

    def get_task_cfg(self, task_name: str):
        """获取任务的默认配置实例。"""
        if task_name not in self._tasks:
            raise ValueError(f"任务 {task_name} 未找到。")
        return self._tasks[task_name].env_cfg_class()

    def make(self, task_name: str, cfg: Any = None, **kwargs):
        """根据任务名和配置创建环境实例。"""
        spec = self._tasks[task_name]
        env = spec.env_class(cfg=cfg, **kwargs)
        return env

    def get_task_names(self):
        return list(self._tasks.keys())

# 全局单例
task_registry = TaskRegistry()
