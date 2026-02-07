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

class Sekiro(ManagerBasedRLEnv):
    """
    只狼任务环境类。
    由配置驱动，基类负责大部分调度工作。
    """
    def __init__(self, cfg: SekiroEnvCfg = None):
        if cfg is None:
            cfg = SekiroEnvCfg()
        super().__init__(cfg)

    def step(self, action):
        """执行一步并返回奖励。"""
        return super().step(action)

    def render_debug(self, seq_np):
        """更新调试可视化输入。"""
        if self.scene_manager:
            self.scene_manager.update_debug_visualization(seq_np)

if __name__ == '__main__':
    # 测试代码
    cfg = SekiroEnvCfg()
    env = Sekiro(cfg)
    print(f"Sekiro task initialized. Action dim: {env.action_dim}")
