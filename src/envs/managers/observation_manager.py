from src.envs.tasks.sekiro.metrics import extract_metrics_from_memory
import configs.config as config

class ObservationManager:
    """
    观测管理器：负责数值状态提取（内存/图像）和状态组装。
    对应 Isaac Lab 中的 ObservationManager。
    """
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
        self.frame_history_len = config.FRAME_HISTORY_LEN
        
        # 内部状态缓存
        self.current_metrics = {
            'self_blood': 0,
            'boss_blood': 0,
            'self_stamina': 0,
            'boss_stamina': 0
        }

    def compute_observations(self):
        """获取最新的数值指标和图像观测。"""
        # 1. 从内存提取指标
        sb, bb, ss, bs = extract_metrics_from_memory()
        
        metrics = {
            'self_blood': sb,
            'boss_blood': bb,
            'self_stamina': ss,
            'boss_stamina': bs
        }
        
        # 2. 从回放缓冲获取堆叠帧（如果需要在此处获取）
        # 注意：在原始代码中，stacked_np 是在 train.py 中获取的，
        # 但在 Isaac Lab 架构中，环境应该返回完整的 obs。
        # 为了兼容性，我们先只管理数值指标。
        
        return metrics

    def get_latest_stacked_frames(self):
        """从缓冲区取最近 k 帧。"""
        return self.replay_buffer.get_latest_observation(self.frame_history_len)
