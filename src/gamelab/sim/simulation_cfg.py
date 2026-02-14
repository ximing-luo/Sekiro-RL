from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class SimulationCfg:
    """仿真环境的基础配置。
    
    对标 Isaac Lab 的 SimulationCfg，针对实时游戏环境进行了适配。
    """
    
    dt: float = 1.0 / 60.0
    """仿真步长（秒）。默认为 60Hz。"""
    
    device: str = "cuda:0"
    """计算设备。默认为 cuda:0。"""
    
    window_title: str = "Sekiro"
    """游戏窗口标题，用于定位和捕获。"""
    
    window_pos: str = "top_left"
    """窗口对齐位置。可选：top_left, top_right, bottom_left, bottom_right, center。"""
    
    headless: bool = False
    """是否以无头模式运行（不显示调试窗口）。"""
    
    use_gpu_pipeline: bool = True
    """是否使用 GPU 加速数据流（如 Vision Tensor 直接在 GPU 上处理）。"""

    debug_vis_fps: int = 0
    """调试可视化窗口的刷新频率 (FPS)。如果 <= 0，则不启动。"""
    
    def validate(self):
        """验证配置合法性。"""
        if self.dt <= 0:
            raise ValueError(f"dt 必须大于 0, 当前为: {self.dt}")
