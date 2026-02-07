import argparse
import sys
import torch
import os
from typing import Any, Dict, Optional

from src.tasks.registration import task_registry

class AppLauncher:
    """应用程序启动器：负责解析命令行参数并初始化环境。
    
    对标 Isaac Lab 的 AppLauncher，统一管理设备、任务、渲染模式等配置。
    """

    def __init__(self, launcher_args: Optional[argparse.Namespace] = None):
        """初始化启动器。
        
        Args:
            launcher_args: 预解析的命令行参数。如果为 None，则尝试解析 sys.argv。
        """
        if launcher_args is None:
            parser = argparse.ArgumentParser(description="Sekiro-RL 应用程序启动器")
            self.add_app_launcher_args(parser)
            self.args, _ = parser.parse_known_args()
        else:
            self.args = launcher_args

        # 设置设备
        self.device = self.args.device
        if "cuda" in self.device and not torch.cuda.is_available():
            print(f"[WARN] CUDA 不可用，回退到 CPU")
            self.device = "cpu"
        
        # 导出给外部使用
        self.task_name = self.args.task
        self.num_envs = self.args.num_envs
        self.headless = self.args.headless

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser):
        """向解析器添加标准启动参数。"""
        arg_group = parser.add_argument_group("app_launcher 核心参数")
        
        arg_group.add_argument(
            "--task", 
            type=str, 
            default="Sekiro-v0", 
            help="要运行的任务名称 (需在 task_registry 中注册)"
        )
        arg_group.add_argument(
            "--num_envs", 
            type=int, 
            default=1, 
            help="并行环境的数量"
        )
        arg_group.add_argument(
            "--device", 
            type=str, 
            default="cuda:0", 
            help="计算设备 (cpu, cuda, cuda:N)"
        )
        arg_group.add_argument(
            "--headless", 
            action="store_true", 
            default=False, 
            help="是否以无头模式运行 (不显示调试窗口)"
        )
        arg_group.add_argument(
            "--seed", 
            type=int, 
            default=None, 
            help="随机种子"
        )
        arg_group.add_argument(
            "--video",
            action="store_true",
            default=False,
            help="是否录制视频"
        )

    def create_env(self, **kwargs) -> Any:
        """根据启动器配置创建环境实例。"""
        print(f"[INFO] 正在创建任务: {self.task_name} (envs={self.num_envs}, device={self.device})")
        
        # 合并参数
        env_kwargs = {
            "num_envs": self.num_envs,
            "device": self.device,
            "headless": self.headless,
        }
        env_kwargs.update(kwargs)
        
        # 从注册表创建
        env, env_cfg = task_registry.make(self.task_name, **env_kwargs)
        return env, env_cfg
