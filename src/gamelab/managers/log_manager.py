import os
import configs.config as config

class LogManager:
    """
    日志管理器：负责收集训练度量。
    目前主要由 TensorBoard 处理，文件日志已移除。
    """
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or config.cfg.path.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
    def log_step(self, agent, env, step, episode, action, reward, recent_rewards, fps, epsilon):
        """
        记录一步的训练数据。
        (当前仅保留接口，具体逻辑移至 TensorBoard 回调)
        """
        pass
