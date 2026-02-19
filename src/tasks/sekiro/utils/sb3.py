import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from .tensorboard import TensorboardHookManager

class SekiroCombinedCallback(BaseCallback):
    """
    全能回调类：
    1. 集成 Isaac Lab 风格打印（含奖励分量）。
    2. 集成卷积层特征图可视化。
    3. 实时特征相似度监控（诊断特征坍缩）。
    """
    def __init__(self, verbose=0, log_interval=2048):
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
        import shutil
        for d in ["logs/data/debug_samples", "logs/data/plots"]:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

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
    
    def _on_rollout_start(self):
        # 在每轮开始时重置环境，确保从头开始
        if self.training_env is not None:
            obs = self.training_env.reset()
            # 同步更新 SB3 Model 的内部状态
            self.model._last_obs = obs

    def _on_rollout_end(self):
        self._analyze_feature_similarity()
        self._plot_value_returns()
        self._plot_returns_timeline()
        # 仿 SB3 风格日志打印 (Feature Analysis)
        # 监控特征相似度：>0.9 意味着特征坍缩风险（红色警告），否则为正常（绿色）
        print("-" * 46)
        print(f"| {'feature_analysis/':<23} | {'':<16} |")
        for key, value in self.last_sim_metrics.items():
            color = "\033[91m" if value > 0.9 else "\033[92m"
            reset = "\033[0m"
            print(f"|    {key:<19} | {color}{value:<16.4f}{reset} |")
        
        # 打印本轮 Reward Components 平均值
        if self.reward_buffer:
            print("-" * 46)
            print(f"| {'reward_components/':<23} | {'Mean':<16} |")
            for key in sorted(self.reward_buffer.keys()):
                values = self.reward_buffer[key]
                if values:
                    avg = np.mean(values)
                    # 绿色正值，红色负值
                    color = "\033[92m" if avg > 0 else "\033[91m" if avg < 0 else "\033[0m"
                    reset = "\033[0m"
                    print(f"|    {key:<19} | {color}{avg:<16.4f}{reset} |")
            # 清空 Buffer 准备下一轮
            self.reward_buffer.clear()
        print("-" * 46)

    def _plot_value_returns(self):
        if not hasattr(self.model, "rollout_buffer"):
            return
            
        save_dir = os.path.join("logs", "data", "plots")
        os.makedirs(save_dir, exist_ok=True)
        
        # Access buffer data
        values = self.model.rollout_buffer.values.flatten()
        returns = self.model.rollout_buffer.returns.flatten()
        
        if len(values) == 0:
            return

        y_true = returns
        y_pred = values
        var_y = np.var(y_true)
        if var_y == 0:
            explained_var = np.nan
        else:
            explained_var = 1 - np.var(y_true - y_pred) / var_y

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.1, s=2)
        
        # Plot ideal line y=x
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        
        plt.xlabel('Returns (True)')
        plt.ylabel('Values (Predicted)')
        plt.title(f'Explained Variance: {explained_var:.4f} (Step {self.num_timesteps})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        save_path = os.path.join(save_dir, f"rollout_{self.num_timesteps}.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_returns_timeline(self):
        if not hasattr(self.model, "rollout_buffer"):
            return
            
        save_dir = os.path.join("logs", "data", "plots")
        os.makedirs(save_dir, exist_ok=True)
        
        # Access buffer data
        # Only plot for the first environment (env_idx=0) to keep it clean
        # rollout_buffer shape: (n_steps, n_envs, ...)
        
        # Check if buffer has data
        if self.model.rollout_buffer.size() == 0:
            return

        # Note: rollout_buffer stores data as (buffer_size, n_envs, ...)
        # We need to access the data directly from the buffer arrays
        returns = self.model.rollout_buffer.returns[:, 0]
        rewards = self.model.rollout_buffer.rewards[:, 0]
        episode_starts = self.model.rollout_buffer.episode_starts[:, 0]
        values = self.model.rollout_buffer.values[:, 0]
        
        n_steps = len(returns)
        if n_steps == 0:
            return
            
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot Returns on left axis (Blue)
        color = 'tab:blue'
        ax1.set_xlabel('Time Step (Rollout)')
        ax1.set_ylabel('Returns (Accumulated)', color=color)
        ax1.plot(returns, color=color, label='Returns', linewidth=1.5, alpha=0.8)
        # Also plot predicted values for comparison
        ax1.plot(values, color='cyan', label='Values (Pred)', linewidth=1.0, linestyle='--', alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot Rewards on right axis (Orange)
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Rewards (Instant)', color=color)
        
        # Use stem plot or bar-like plot for sparse rewards, but plot is easier for dense
        ax2.plot(rewards, color=color, label='Rewards', linewidth=1.0, alpha=0.5)
        
        # Auto-scale Y-axis for rewards to take up ~2/3 of the plot height and center it
        r_min, r_max = np.min(rewards), np.max(rewards)
        if r_min != r_max:
            # Add small padding to avoid min/max touching the edge of the scaled region
            r_range = r_max - r_min
            # We want the data to occupy 2/3 of the total vertical space
            # Total_Span * (2/3) = Data_Range
            # Total_Span = Data_Range * 1.5
            total_span = r_range * 1.5
            mid_point = (r_max + r_min) / 2
            ax2.set_ylim(mid_point - total_span/2, mid_point + total_span/2)
            
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Mark episode starts with dark red vertical lines
        starts = np.where(episode_starts)[0]
        for start in starts:
            ax1.axvline(x=start, color='darkred', linestyle='-', linewidth=1.5, alpha=0.8)
            
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'Returns & Rewards Timeline (Env 0, Step {self.num_timesteps})')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, f"timeline_{self.num_timesteps}.png")
        plt.savefig(save_path)
        plt.close()

    def _analyze_feature_similarity(self):
        device = self.model.device
        extractor = self.model.policy.features_extractor
        extractor.eval()
        
        # 获取遥测数据的维度
        tele_dim = self.model.observation_space['telemetry'].shape[0]
        
        with torch.no_grad():
            # 1. 构造假数据 (Synthetic Inputs)
            # Focus 层要求输入宽高为偶数，使用 136x240 避免尺寸不匹配
            noise_img = torch.randn(1, 3, 136, 240).to(device)
            black_img = torch.zeros(1, 3, 136, 240).to(device)
            checker_img = torch.ones(1, 3, 136, 240).to(device)
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
                num_samples = min(64, flat_img.size(0))
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
