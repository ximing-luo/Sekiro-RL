import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from .tensorboard import TensorboardHookManager

class SekiroCombinedCallback(BaseCallback):
    """
    全能回调类：
    1. 集成 Isaac Lab 风格打印（含奖励分量）。
    2. 集成卷积层特征图可视化。
    3. 实时特征相似度监控（诊断特征坍缩）。
    """
    def __init__(self, verbose=0, log_interval=1000):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.hook_manager = None
        self.iteration = 0
        self.reward_buffer = {}
        self.cum_player_deaths = 0
        self.cum_enemy_deaths = 0
        self.last_sim_metrics = {
            "synthetic_avg": 0.0,
            "experience_avg": 0.0
        }
        print("SekiroCombined Callback Initialized")

    def _on_training_start(self):
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        writer = None
        if self.model.logger is not None:
            for output_format in self.model.logger.output_formats:
                if isinstance(output_format, TensorBoardOutputFormat):
                    writer = output_format.writer
                    break
        
        if writer is not None:
            self.hook_manager = TensorboardHookManager(self.model, writer, log_interval=self.log_interval)
            self.hook_manager.register_hooks()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" in info:
                components = info["reward_components"]
                for name, val in components.items():
                    if name not in self.reward_buffer:
                        self.reward_buffer[name] = []
                    self.reward_buffer[name].append(val)
            if "events" in info:
                events = info["events"]
                if 0 in events: 
                    self.cum_player_deaths += 1
                if 1 in events: 
                    self.cum_enemy_deaths += 1
        return True

    def _on_rollout_end(self):
        self._analyze_feature_similarity()
        # 仿 SB3 风格日志打印 (Feature Analysis)
        # 监控特征相似度：>0.9 意味着特征坍缩风险（红色警告），否则为正常（绿色）
        print("-" * 46)
        print(f"| {'feature_analysis/':<23} | {'':<16} |")
        for key, value in self.last_sim_metrics.items():
            color = "\033[91m" if value > 0.9 else "\033[92m"
            reset = "\033[0m"
            print(f"|    {key:<19} | {color}{value:<16.4f}{reset} |")
        print("-" * 46)

    def _analyze_feature_similarity(self):
        device = self.model.device
        extractor = self.model.policy.features_extractor
        extractor.eval()
        
        # 获取遥测数据的维度
        tele_dim = self.model.observation_space['telemetry'].shape[0]
        
        with torch.no_grad():
            # 1. 构造假数据 (Synthetic Inputs)
            noise_img = torch.randn(1, 3, 135, 240).to(device)
            black_img = torch.zeros(1, 3, 135, 240).to(device)
            checker_img = torch.ones(1, 3, 135, 240).to(device)
            checker_img[:, :, ::2, ::2] = 0
            
            # 构造对应的假遥测数据 (全0)
            dummy_tele = torch.zeros(1, tele_dim).to(device)
            
            syn_inputs = [
                {'policy': noise_img, 'telemetry': dummy_tele},
                {'policy': black_img, 'telemetry': dummy_tele},
                {'policy': checker_img, 'telemetry': dummy_tele}
            ]
            
            syn_feats = [extractor(inp) for inp in syn_inputs]
            syn_sims = []
            for i in range(len(syn_feats)):
                for j in range(i + 1, len(syn_feats)):
                    sim = F.cosine_similarity(syn_feats[i], syn_feats[j]).item()
                    syn_sims.append(sim)
            self.last_sim_metrics["synthetic_avg"] = sum(syn_sims) / len(syn_sims) if syn_sims else 1.0

            # 2. 从 Rollout Buffer 采样真实数据
            if hasattr(self.model, "rollout_buffer") and self.num_timesteps > 0:
                obs_dict = self.model.rollout_buffer.observations
                
                # 处理图像数据 "policy"
                img_obs = obs_dict['policy'] # (n_steps, n_envs, C, H, W)
                flat_img = torch.as_tensor(img_obs).view(-1, *img_obs.shape[2:]) # (N, C, H, W)
                # 处理遥测数据 "telemetry"
                tele_obs = obs_dict['telemetry'] # (n_steps, n_envs, tele_dim)
                flat_tele = torch.as_tensor(tele_obs).view(-1, *tele_obs.shape[2:]) # (N, tele_dim)
                # 随机采样
                num_samples = min(8, flat_img.size(0))
                idx = torch.randperm(flat_img.size(0))[:num_samples]
                
                # 采样原始数据 (uint8)
                batch_img = flat_img[idx].to(device)
                batch_tele = flat_tele[idx].to(device)
                
                # 保存调试图像 (传入原始 uint8 数据，避免 float 截断导致全黑)
                self._save_debug_images(batch_img)
                
                # 构造 Batch Input
                exp_samples = {
                    'policy': batch_img,
                    'telemetry': batch_tele
                }
                
                # 提取特征
                exp_feats = extractor(exp_samples)
                
                # 计算相似度矩阵
                norm_exp_feats = F.normalize(exp_feats, dim=1)
                sim_matrix = torch.matmul(norm_exp_feats, norm_exp_feats.T)
                n = exp_feats.size(0)
                mask = torch.eye(n, device=device).bool()
                exp_sims = sim_matrix[~mask]
                if exp_sims.numel() > 0:
                    self.last_sim_metrics["experience_avg"] = exp_sims.mean().item()
        extractor.train()

    def _save_debug_images(self, samples):
        debug_dir = os.path.join("logs", "data", "debug_samples")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        for i in range(samples.size(0)):
            img = samples[i].cpu().numpy().transpose(1, 2, 0)
            img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, f"sample_{i}.png"), img)
