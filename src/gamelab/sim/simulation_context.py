

import time
import torch
import weakref
from typing import Dict, Optional, Any

from .simulation_cfg import SimulationCfg
from src.gamelab.sim.sensors.base_sensor import BaseSensor
import src.gamelab.interfaces.window_utils as window_utils

class SimulationContext:
    """仿真上下文管理器：负责游戏环境的生命周期、窗口同步与传感器调度。
    
    对标 Isaac Lab 的 SimulationContext，采用单例模式。
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SimulationContext, cls).__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls) -> "SimulationContext":
        """获取单例实例。"""
        if cls._instance is None:
            raise RuntimeError("SimulationContext 尚未初始化，请先调用构造函数。")
        return cls._instance

    def __init__(self, cfg: Optional[SimulationCfg] = None):
        """初始化仿真上下文。
        
        Args:
            cfg: 仿真配置。如果为 None，则使用默认配置。
        """
        # 防止重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        if cfg is None:
            cfg = SimulationCfg()
        cfg.validate()
        self.cfg = cfg
        
        self.device = torch.device(self.cfg.device)
        self.sensors: Dict[str, BaseSensor] = {}
        self.is_running = False
        
        # 记录仿真时间
        self._sim_time = 0.0
        self._step_count = 0
        
        self._initialized = True
        print(f"[INFO] SimulationContext 已初始化 (device={self.device}, dt={self.cfg.dt})")

    def add_sensor(self, name: str, sensor: BaseSensor):
        """注册传感器。"""
        self.sensors[name] = sensor

    def setup(self):
        """启动仿真环境。"""
        print(f"[INFO] 正在设置仿真环境: {self.cfg.window_title}...")
        
        # 1. 窗口定位
        window_utils.move_window(self.cfg.window_title, self.cfg.window_pos, True)
        window_utils.activate_window_by_title_contains(self.cfg.window_title, True)
        
        # 2. 启动传感器
        for name, sensor in self.sensors.items():
            print(f"[INFO] 启动传感器: {name}")
            sensor.start()
            
        # 3. 预热
        time.sleep(1.0)
        self.is_running = True
        print("[INFO] 仿真环境已就绪。")

    def step(self, render: bool = True):
        """执行一个仿真步。
        
        在实时游戏中，这主要用于同步传感器数据并更新仿真时间。
        """
        if not self.is_running:
            return

        # 更新时间戳（即使是实时游戏，我们也维持一个逻辑时钟）
        self._sim_time += self.cfg.dt
        self._step_count += 1
        
        # 这里可以加入强制同步逻辑（如果需要对齐游戏帧）

    def stop(self):
        """停止仿真并清理资源。"""
        for sensor in self.sensors.values():
            sensor.stop()
        self.is_running = False
        print("[INFO] 仿真环境已停止。")

    @property
    def sim_time(self) -> float:
        """当前的逻辑仿真时间。"""
        return self._sim_time

    @property
    def step_count(self) -> int:
        """已执行的仿真步数。"""
        return self._step_count
