from .base_sensor import BaseSensor
from typing import Any

class MemorySensor(BaseSensor):
    """
    内存传感器：封装遥测驱动。
    """
    def __init__(self, telemetry_impl):
        self.telemetry = telemetry_impl

    def start(self):
        if hasattr(self.telemetry, 'start'):
            self.telemetry.start()

    def stop(self):
        if hasattr(self.telemetry, 'stop'):
            self.telemetry.stop()

    def get_data(self) -> Any:
        if hasattr(self.telemetry, 'get_metrics'):
            return self.telemetry.get_metrics()
        return {}
