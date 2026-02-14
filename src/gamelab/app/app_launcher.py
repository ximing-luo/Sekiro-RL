import argparse
import torch
from typing import Any, Optional

from src.tasks.registration import task_registry
import src.tasks # 确保所有任务都被注册

class AppLauncher:
    """应用程序启动器：负责解析命令行参数并初始化环境。
    
    对标 Isaac Lab 的 AppLauncher，统一管理设备、任务、渲染模式等配置。
    """

    def __init__(self, launcher_args: Optional[argparse.Namespace] = None):
        """初始化启动器。
        
        基于“执行必然性”原则：设备设置应一次性确定。
        """
        if launcher_args is None:
            parser = argparse.ArgumentParser(description="Sekiro-RL 应用程序启动器")
            self.add_app_launcher_args(parser)
            self.args, _ = parser.parse_known_args()
        else:
            self.args = launcher_args

        # 确立设备契约
        self.device = self.args.device
        if "cuda" in self.device and not torch.cuda.is_available():
            raise RuntimeError(f"[AppLauncher] CUDA 请求失败：设备 {self.device} 不可用。")
        
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
