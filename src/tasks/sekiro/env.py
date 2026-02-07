"""
模块用途：重写后的 Sekiro 环境任务类。
继承自基类 ManagerBasedRLEnv，实现极致的逻辑抽离。
"""
import os
import sys

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gamelab.envs.manager_based_rl_env import ManagerBasedRLEnv
from src.tasks.sekiro.sekiro_env_cfg import SekiroEnvCfg
from src.gamelab.sim.sensors.memory_sensor import MemorySensor
from src.gamelab.assets.sekiro.telemetry import SekiroTelemetry

class Sekiro(ManagerBasedRLEnv):
    """
    只狼任务环境类。
    对标 Isaac Lab 的 Task 模式，负责组装特定的传感器。
    """
    def __init__(self, cfg: SekiroEnvCfg = None):
        if cfg is None:
            cfg = SekiroEnvCfg()
        super().__init__(cfg)

    def _setup_managers(self):
        """扩展基类的初始化。"""
        # 调用基类完成通用初始化 (如 VisionSensor, InteractiveScene assets)
        super()._setup_managers()

    def step(self, action):
        """执行一步并返回奖励。"""
        return super().step(action)

    def render_debug(self, seq_np):
        """更新调试可视化输入。"""
        if hasattr(self, '_input_vis_runner') and self._input_vis_runner:
            self._input_vis_runner.update(seq_np)

if __name__ == '__main__':
    # 测试代码
    cfg = SekiroEnvCfg()
    env = Sekiro(cfg)
    print(f"Sekiro task initialized. Action dim: {env.action_dim}")
