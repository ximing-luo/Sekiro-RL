import json
import os
import time

class RecorderManager:
    """
    记录管理器：用于录制专家演示数据（Demonstrations）。
    为模仿学习 (Mimicry) 提供训练素材。
    """
    def __init__(self, save_dir="data/demos"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.current_trajectory = []

    def record_step(self, obs, action, metrics):
        """记录一步的数据。"""
        step_data = {
            "timestamp": time.time(),
            "action": action,
            "metrics": metrics
            # 注意：图像数据通常很大，建议只记录关键指标或单独保存视频
        }
        self.current_trajectory.append(step_data)

    def save_trajectory(self, filename=None):
        """保存当前轨迹。"""
        if not self.current_trajectory:
            return
        
        if filename is None:
            filename = f"demo_{int(time.time())}.json"
            
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.current_trajectory, f, indent=2)
            
        print(f"Trajectory saved to: {path}")
        self.current_trajectory = []
