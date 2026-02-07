import torch as th

def null_command(params: dict = None):
    """不执行任何操作的空指令。"""
    return th.zeros(1)

def uniform_velocity_command(params: dict):
    """
    均匀分布的随机速度指令示例。
    """
    low = params.get("low", 0.0)
    high = params.get("high", 1.0)
    return th.empty(1).uniform_(low, high)
