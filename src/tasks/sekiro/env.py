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

from src.envs.manager_based_rl_env import ManagerBasedRLEnv
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
        
        # 兼容旧代码需要的属性
        self.width = cfg.scene.observation_w
        self.height = cfg.scene.observation_h
        self.n_step_rewards = cfg.n_step_rewards
        
        # 兼容旧代码直接访问指标的需求
        self.self_blood = 0
        self.boss_blood = 0
        self.self_stamina = 0
        self.boss_stamina = 0

    def step(self, action):
        """执行一步并更新兼容性指标。"""
        reward = super().step(action)
        
        # 更新兼容性指标
        if self.last_metrics:
            self.self_blood = self.last_metrics['self_blood']
            self.boss_blood = self.last_metrics['boss_blood']
            self.self_stamina = self.last_metrics['self_stamina']
            self.boss_stamina = self.last_metrics['boss_stamina']
            
        return reward

    def update_debug_visual_input(self, seq_np):
        """兼容接口：更新调试可视化输入。"""
        if self.scene_manager:
            self.scene_manager.update_debug_visualization(seq_np)

    @property
    def action_dim(self):
        """兼容接口：获取动作空间维度。"""
        return self.action_manager.get_action_dim()

if __name__ == '__main__':
    # 测试代码
    cfg = SekiroEnvCfg()
    env = Sekiro(cfg)
    print(f"Sekiro task initialized. Action dim: {env.action_dim}")
