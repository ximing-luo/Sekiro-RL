import time
from .manager_based_env_cfg import ManagerBasedEnvCfg
from src.scene import SceneManager
from src.managers import (
    ActionManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
    LogManager
)

class ManagerBasedEnv:
    """
    基础管理器驱动环境类。
    负责协调场景和各个管理器的生命周期。
    """
    def __init__(self, cfg: ManagerBasedEnvCfg):
        self.cfg = cfg
        
        # 1. 初始化场景管理器 (物理底座)
        # 注意：这里假设子类或配置会提供 replay_buffer
        self.replay_buffer = None # 由子类初始化
        self.scene_manager: Optional[SceneManager] = None
        
        # 2. 初始化逻辑管理器
        self.action_manager: Optional[ActionManager] = None
        self.observation_manager: Optional[ObservationManager] = None
        self.reward_manager: Optional[RewardManager] = None
        self.termination_manager: Optional[TerminationManager] = None
        self.log_manager: Optional[LogManager] = None

    def _setup_managers(self):
        """由子类调用以完成管理器的实例化。"""
        scene_cfg = self.cfg.scene
        self.scene_manager = SceneManager(
            scene_cfg.observation_w, 
            scene_cfg.observation_h, 
            self.replay_buffer, 
            scene_cfg.pos, 
            scene_cfg.capture_fps
        )
        self.scene_manager.setup()
        
        if scene_cfg.debug_vis_fps > 0:
            self.scene_manager.start_debug_visualization(scene_cfg.debug_vis_fps)

    def pause_game(self, paused):
        """处理暂停逻辑。"""
        should_over, paused = self.scene_manager.check_pause(paused)
        return should_over, paused

    def close(self):
        """清理资源。"""
        if self.scene_manager:
            self.scene_manager.stop()
