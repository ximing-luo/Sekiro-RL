# Copyright (c) 2024, Sekiro-RL Project.
# All rights reserved.

import os
import torch
import torch.nn as nn
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from src.framework.sb3.ppo_aux import AuxPPO
from src.models.ppo_models import SekiroMultiInputExtractor
from src.gamelab.utils.rl.sb3 import SekiroCombinedCallback
from src.utils.import_utils import filter_kwargs
import configs.config as config

class SB3OnPolicyRunner:
    """对标 IsaacLab OnPolicyRunner 的 SB3 封装。
    
    统一管理模型初始化、日志配置、回调挂载和训练循环。
    """
    
    def __init__(self, env, args, log_dir, device="cuda:0"):
        self.env = env
        self.args = args
        self.log_dir = log_dir
        self.device = device
        
        # 1. 配置日志
        self.logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        
        # 2. 准备模型参数
        self.policy_kwargs = dict(
            features_extractor_class=SekiroMultiInputExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[512, 256], vf=[512, 256]),
            activation_fn=nn.ReLU
        )
        
        # 3. 初始化模型
        self.ppo_kwargs = filter_kwargs(AuxPPO, vars(self.args))
        self._setup_model()
        
        # 4. 挂载回调
        self._setup_callbacks()

    def _setup_model(self):
        checkpoint_path = self.args.checkpoint
        if self.args.resume and checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] 正在从断点加载模型: {checkpoint_path}")
            self.model = AuxPPO.load(
                checkpoint_path, 
                env=self.env, 
                device=self.device,
                custom_objects=self.ppo_kwargs
            )
        else:
            print(f"[INFO] 正在初始化新模型 (设备: {self.device})...")
            self.model = AuxPPO(
                "MultiInputPolicy",
                self.env, 
                verbose=1, # 日志打印级别:0-无输出, 1-进度表, 2-调试
                policy_kwargs=self.policy_kwargs,
                device=self.device,
                **self.ppo_kwargs
            )
        self.model.set_logger(self.logger)

    def _setup_callbacks(self):
        self.callbacks = []
        
        # 保存断点
        self.callbacks.append(CheckpointCallback(
            save_freq=self.args.save_freq,
            save_path=os.path.join(self.log_dir, "checkpoints"),
            name_prefix="sekiro_ppo"
        ))
        
        # Sekiro 专用监控
        self.callbacks.append(SekiroCombinedCallback(
            log_interval=config.cfg.path.tb_log_interval
        ))

    def learn(self, total_timesteps=None):
        steps = total_timesteps if total_timesteps else self.args.steps
        print(f"[INFO] 训练开始。目标总步数: {steps}")
        try:
            self.model.learn(
                total_timesteps=steps, 
                callback=self.callbacks,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("[WARN] 训练被手动中断。")
        finally:
            self.save("final")

    def save(self, name):
        save_path = os.path.join(self.log_dir, "checkpoints", f"sekiro_ppo_{name}")
        self.model.save(save_path)
        # 如果是 VecNormalize 环境，保存统计量
        if hasattr(self.env, "save"):
            self.env.save(os.path.join(self.log_dir, "checkpoints", f"sekiro_vec_normalize_{name}.pkl"))
        print(f"[INFO] 模型已保存至: {save_path}")
