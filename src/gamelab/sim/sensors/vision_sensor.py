from .base_sensor import BaseSensor
from src.gamelab.interfaces.capture.driver import FrameCapture
import numpy as np

class VisionSensor(BaseSensor):
    """
    视觉传感器：封装画面采集驱动。
    """
    def __init__(self, camera_index, camera_width, camera_height, fps, target_width, target_height):
        self.driver = FrameCapture(
            camera_index=camera_index,
            camera_width=camera_width,
            camera_height=camera_height,
            fps=fps,
            target_width=target_width,
            target_height=target_height
        )

    def start(self):
        self.driver.start()

    def stop(self):
        self.driver.stop()

    def get_data(self) -> np.ndarray:
        return self.driver.latest_frame
