from dataclasses import dataclass
from src.envs.base.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

@dataclass
class ManagerBasedRLMimicEnvCfg(ManagerBasedRLEnvCfg):
    demo_path: str = "" # 专家轨迹数据路径
    mimic_weight: float = 0.5 # 模仿奖励权重
