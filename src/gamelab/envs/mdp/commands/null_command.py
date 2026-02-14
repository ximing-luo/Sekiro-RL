import torch as th

def null_command(env_ids=None, cfg=None):
    """不执行任何操作的空指令。"""
    num_envs = len(env_ids) if env_ids is not None else 1
    # 动态获取形状，如果 cfg 没定义则默认为 (1,)
    shape = getattr(cfg, "shape", (1,))
    return th.zeros((num_envs, *shape))

def uniform_velocity_command(env_ids=None, cfg=None):
    """
    均匀分布的随机速度指令示例。
    """
    num_envs = len(env_ids) if env_ids is not None else 1
    shape = getattr(cfg, "shape", (1,))
    low = getattr(cfg, "low", 0.0)
    high = getattr(cfg, "high", 1.0)
    return th.empty((num_envs, *shape)).uniform_(low, high)
