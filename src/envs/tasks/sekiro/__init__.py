from src.envs.tasks.registration import task_registry
from .env import Sekiro
from .sekiro_env_cfg import SekiroEnvCfg

# 注册任务到全局注册中心
task_registry.register(
    task_name="Sekiro-v0",
    env_class=Sekiro,
    env_cfg_class=SekiroEnvCfg
)
