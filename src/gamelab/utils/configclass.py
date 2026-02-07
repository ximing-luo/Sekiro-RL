from dataclasses import dataclass

def configclass(cls):
    """将类转换为增强的 dataclass，用于配置管理。
    对标 Isaac Lab 的 configclass。
    """
    # 默认设置 frozen=False, repr=True
    return dataclass(cls, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
