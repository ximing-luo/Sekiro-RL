"""
通用原子观测函数库。
"""

from typing import Any, Dict
import torch as th

def sensor_data(env, sensor_name: str, **kwargs) -> Any:
    """
    直接获取指定传感器的原始数据。
    """
    sensor = env.sim.get_sensor(sensor_name)
    return sensor.get_data() if sensor else None

def sensor_data_flattened(env, sensor_name: str, keys: list = None, **kwargs) -> th.Tensor:
    """
    获取传感器数据并展平为 Tensor。
    """
    sensor = env.sim.get_sensor(sensor_name)
    if not sensor:
        return th.zeros(1)
    
    data = sensor.get_data()
    if keys:
        values = [data.get(k, 0) for k in keys]
    else:
        values = list(data.values())
        
    return th.tensor(values, dtype=th.float32)
