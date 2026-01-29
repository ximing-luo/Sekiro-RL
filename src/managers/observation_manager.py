from src.envs.mdp.observations import extract_metrics_from_memory
import configs.config as config

class ObservationManager:
    """
    观测管理器：负责数值状态提取（内存/图像）和状态组装。
    对应 Isaac Lab 中的 ObservationManager。
    """
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
        self.frame_history_len = config.FRAME_HISTORY_LEN

    def compute_observations(self, scene_manager=None):
        """获取最新的数值指标和图像观测。"""
        # 1. 从内存提取指标
        sb, bb, ss, bs = extract_metrics_from_memory()
        
        metrics = {
            'self_blood': sb,
            'boss_blood': bb,
            'self_stamina': ss,
            'boss_stamina': bs
        }
        
        # 2. 获取最新图像帧
        frame = None
        if scene_manager:
            frame = scene_manager.get_latest_frame()
            metrics['frame'] = frame
        
        return metrics

    def get_latest_stacked_frames(self):
        """从缓冲区取最近 k 帧。"""
        return self.replay_buffer.get_latest_observation(self.frame_history_len)
