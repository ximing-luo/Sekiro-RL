import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple, Generator, Optional, Union, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutBufferSamples
from .buffer import SekiroRolloutBuffer

class SekiroPPO(PPO):
    """
    集成 VICReg 和逆动力学辅助任务的自定义 PPO 算法。
    
    1. VICReg Variance Loss: 强制特征向量分化。
    2. Inverse Dynamics Loss: 根据特征预测动作，强化动态特征提取。
    """
    def __init__(self, *args, vicreg_coef=0.1, inv_dyn_coef=0.1, clip_range_vf=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vicreg_coef = vicreg_coef
        self.inv_dyn_coef = inv_dyn_coef
        self.clip_range_vf = clip_range_vf
        
        # 显存优化：手动设置 RolloutBuffer 类别
        self.rollout_buffer_class = SekiroRolloutBuffer
        
        # 逆动力学预测头：输入两个特征向量 (z_t, z_{t+1})，预测动作 a_t
        features_dim = 512
        self.inv_dyn_head = nn.Sequential(
            nn.Linear(features_dim * 2, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, self.action_space.n)
        ).to(self.device)
        
        # 将预测头参数加入优化器
        self.policy.optimizer.add_param_group({'params': self.inv_dyn_head.parameters()})

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # 1. 创建显存优化型 Buffer (使用原始 3 通道空间)
        self.rollout_buffer = SekiroRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        # 2. 创建 Policy (使用 12 通道空间，满足 MADS 提取器要求)
        from gymnasium import spaces
        policy_obs_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(12, self.observation_space.shape[1], self.observation_space.shape[2]),
            dtype=np.uint8
        )
        
        self.policy = self.policy_class(
            policy_obs_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # 3. 初始化最后观测值
        # 注意：这里需要重置环境获取初始观测
        self._last_obs = self.env.reset()
        self._last_episode_starts = np.ones((self.n_envs,), dtype=bool)

    def collect_rollouts(
        self,
        env: GymEnv,
        callback: MaybeCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        重写采集逻辑：
        1. 使用 observation_manager 获取堆叠帧供 Policy 使用。
        2. 向 Buffer 存入单帧和下一帧，实现显存优化。
        """
        assert self._last_obs is not None, "No previous observation"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # --- 1. 获取堆叠帧作为 Policy 输入 ---
                # 使用 SB3 标准方法 get_attr 跨越 VecEnv 包装器获取 observation_manager
                stacked_obs = None
                try:
                    managers = env.get_attr("observation_manager")
                    stacked_obs = managers[0].get_latest_stacked_frames()
                except (AttributeError, IndexError):
                    pass
                
                # 如果获取失败或返回空，尝试从 unwrapped 获取
                if stacked_obs is None:
                    if hasattr(env.unwrapped, "observation_manager"):
                        stacked_obs = env.unwrapped.observation_manager.get_latest_stacked_frames()
                
                # 最后的保底方案：使用最后一次观测并确保通道数匹配
                if stacked_obs is None:
                    stacked_obs = self._last_obs
                
                # 转换为 Tensor 并确保是 4D (Batch, Channel, H, W)
                obs_tensor = torch.as_tensor(stacked_obs).to(self.device)
                
                # 维度检查与对齐
                if obs_tensor.ndim == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
                elif obs_tensor.ndim == 5:
                    obs_tensor = obs_tensor.squeeze(1)
                
                # 通道数对齐：如果还是单帧 (3通道)，重复 4 次以匹配 MADS 的 12 通道输入
                if obs_tensor.shape[1] == 3:
                    obs_tensor = obs_tensor.repeat(1, 4, 1, 1)
                
                actions, values, log_probs = self.policy(obs_tensor)
            
            # 显存优化：将张量从 GPU 转移到 CPU 并脱离计算图
            # 必须保持为 Tensor 类型，因为 SB3 的 RolloutBuffer.add 会调用 .clone()
            # 否则会报 AttributeError: 'numpy.ndarray' object has no attribute 'clone'
            actions = actions.cpu().numpy()
            values = values.flatten().detach().cpu()
            log_probs = log_probs.flatten().detach().cpu()

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            n_steps += 1

            # --- 2. 存入 Buffer (单帧 + 下一帧) ---
            # 确保存入的是单帧 (3, H, W)
            def to_single_frame(o):
                if isinstance(o, np.ndarray):
                    if o.ndim == 4: return o[0, -3:] # 取最后一帧
                    if o.ndim == 3 and o.shape[0] > 3: return o[-3:]
                return o

            rollout_buffer.add(
                to_single_frame(self._last_obs), 
                actions, 
                rewards, 
                self._last_episode_starts, 
                values, 
                log_probs,
                next_obs=to_single_frame(new_obs)
            )
            
            self._last_obs = new_obs
            self._last_episode_starts = dones

            # 更新回调本地变量并执行 step 回调
            callback.update_locals(locals())
            if not callback.on_step():
                return False

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        重写训练逻辑：
        1. 修复 n_epochs 循环。
        2. 优化特征提取，避免双倍显存占用。
        3. 集成辅助任务 Loss。
        4. 增加全量梯度与参数监控。
        """
        self._update_learning_rate(self.policy.optimizer)

        # 初始化内部训练步数计数器 (用于 logging 触发)
        if not hasattr(self, "_train_step_count"):
            self._train_step_count = 0

        # 用于记录平均 Loss
        pg_losses, value_losses, vicreg_losses, inv_dyn_losses = [], [], [], []

        # 进度监控初始化
        n_batches = (self.n_steps * self.n_envs) // self.batch_size
        total_iters = self.n_epochs * n_batches
        current_iter = 0
        print(f"\033[94m[TRAIN] 开始执行反向传播: {self.n_epochs} Epochs | {n_batches} Batches/Epoch | 总计 {total_iters} 次更新\033[0m")

        # --- 反向传播开始前：停止 SceneManager 采集以节省 GPU 资源 ---
        try:
            # 尝试通过 get_attr 获取 VecEnv 中的 scene_manager
            scene_managers = self.env.get_attr("scene_manager")
            if scene_managers and scene_managers[0] is not None:
                scene_managers[0].stop()
                print("\033[93m[INFO] 已停止 SceneManager 采集预览以节省 GPU 资源\033[0m")
        except (AttributeError, IndexError):
            # 如果失败，尝试直接访问 (针对非 VecEnv)
            if hasattr(self.env, "scene_manager"):
                self.env.scene_manager.stop()
                print("\033[93m[INFO] 已停止 SceneManager 采集预览以节省 GPU 资源\033[0m")

        for epoch in range(self.n_epochs):
            # --- 显存预警与优化 ---
            # 每一轮 epoch 都打乱一次 buffer 数据并分批加载。
            # 注意：这里的 self.batch_size 是 mini-batch 大小。
            # 对于 12 通道图像 (12, 135, 240)，单样本约 0.37MB (float32)。
            # batch_size=256 时，obs + next_obs 占用约 190MB 显存。
            # batch_size=4096 时，将直接占用约 3GB 显存，加上中间激活值极易爆显存。
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()
                
                # --- 0. 数值安全检查 ---
                observations = rollout_data.observations
                next_observations = rollout_data.next_observations
                
                # --- 1. 核心前向传播 (优化版：CNN 仅运行一次) ---
                # 提取特征
                features = self.policy.features_extractor(observations)

                # 获取策略和价值分布 (复用 features)
                latent_pi, latent_vf = self.policy.mlp_extractor(features)
                distribution = self.policy._get_action_dist_from_latent(latent_pi)
                log_prob = distribution.log_prob(actions)
                values = self.policy.value_net(latent_vf).flatten()
                entropy = distribution.entropy()
                
                # --- 2. 辅助任务：VICReg Variance Loss ---
                # 强化版：防止由于特征塌缩导致的梯度奇点
                # 1. 注入极小噪声破坏全零状态
                safe_features = features + torch.randn_like(features) * 1e-6
                # 2. 提升 eps 保护，确保反向传播导数不爆炸
                std_features = torch.sqrt(safe_features.var(dim=0) + 1e-03)
                # 3. 计算 Loss 并应用上界裁剪
                vicreg_loss = torch.mean(F.relu(1.0 - std_features))
                vicreg_loss = torch.clamp(vicreg_loss, max=10.0)

                # --- 3. 辅助任务：Inverse Dynamics Loss ---
                # 预测动作：根据 z_t 和 z_{t+1} 预测 a_t
                next_features = self.policy.features_extractor(next_observations)
                pred_actions = self.inv_dyn_head(torch.cat([features, next_features], dim=1))
                # 引入 label_smoothing 以增加分类损失的鲁棒性
                inv_dynamics_loss = F.cross_entropy(pred_actions, actions, label_smoothing=0.1)
                
                # --- 4. PPO 主损失计算 ---
                advantages = rollout_data.advantages
                # 优势归一化
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 获取当前的剪切范围
                if callable(self.clip_range):
                    current_clip_range = self.clip_range(self._current_progress_remaining)
                else:
                    current_clip_range = self.clip_range

                # 策略损失
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - current_clip_range, 1 + current_clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # 价值损失 (包含 Clipping 逻辑以增加稳定性)
                if self.clip_range_vf is not None:
                    if callable(self.clip_range_vf):
                        current_clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
                    else:
                        current_clip_range_vf = self.clip_range_vf
                    
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -current_clip_range_vf, current_clip_range_vf
                    )
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                else:
                    value_loss = F.mse_loss(rollout_data.returns, values)

                # 熵损失
                entropy_loss = -torch.mean(entropy)

                # 总损失计算
                # 注释：所有损失项均为正数（或需要最小化的量），因此使用加号。
                # 1. policy_loss: 策略梯度损失 (已包含负号，最小化 -log_prob * adv)
                # 2. entropy_loss: 熵惩罚 (最小化 -entropy = 最大化熵)
                # 3. value_loss: 价值误差 (MSE，最小化预测偏差)
                # 4. vicreg_loss: 方差正则项 (最小化 relu(1-std) = 强制标准差 >= 1)
                # 5. inv_dyn_loss: 逆动力学损失 (最小化动作预测误差)
                loss = (
                    policy_loss 
                    + self.ent_coef * entropy_loss 
                    + self.vf_coef * value_loss 
                    + self.vicreg_coef * vicreg_loss
                    + self.inv_dyn_coef * inv_dynamics_loss
                )

                # Loss 异常监测 (石锤 3)
                if loss.item() > 1000.0 or torch.isnan(loss):
                    print(f"\n\033[91;1m[LOSS WATCHDOG] 检测到异常 Loss! Step: {self._train_step_count}\033[0m")
                    print(f"  > Total Loss: {loss.item():.4f}")
                    print(f"  > Policy Loss: {policy_loss.item():.4f}")
                    print(f"  > Value Loss: {value_loss.item():.4f}")
                    print(f"  > VICReg Loss: {vicreg_loss.item():.4f}")
                    print(f"  > InvDyn Loss: {inv_dynamics_loss.item():.4f}")
                    if torch.isnan(loss):
                        raise RuntimeError("Loss 变为 NaN")

                # 优化步骤
                self.policy.optimizer.zero_grad()
                try:
                    loss.backward()
                except RuntimeError as e:
                    if "NaN" in str(e):
                        print("\033[91m[FATAL] Backward 传播过程中产生 NaN!\033[0m")
                        print(f"Loss 状态: Policy={policy_loss.item():.4f}, Value={value_loss.item():.4f}, "
                              f"VICReg={vicreg_loss.item():.4f}, InvDyn={inv_dynamics_loss.item():.4f}")
                    raise e
                
                # --- 诊断：调试完成后不再频繁检查梯度与权重以提升性能 ---
                # if hasattr(self, "diagnostics"):
                #     try:
                #         self.diagnostics.log_gradients(step=self._train_step_count)
                #     except RuntimeError as e:
                #         # 记录导致 NaN 的具体 Loss 项 (石锤 2)
                #         print(f"\033[91m[FATAL] 梯度爆炸触发时 Loss 状态:\033[0m")
                #         print(f"  > policy_loss: {policy_loss.item():.6f}")
                #         print(f"  > value_loss:  {value_loss.item():.6f}")
                #         print(f"  > vicreg_loss: {vicreg_loss.item():.6f}")
                #         print(f"  > inv_dyn_loss: {inv_dynamics_loss.item():.6f}")
                #         raise e
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                # --- 进度实时刷新 ---
                current_iter += 1
                progress = current_iter / total_iters
                bar_len = 20
                filled_len = int(bar_len * progress)
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                print(f"\n\033[96m  进度: |{bar}| {progress:6.1%} | Epoch: {epoch+1}/{self.n_epochs} | Loss: {loss.item():.4f}\033[0m", end="")
                
                # --- 诊断：记录更新后的权重 (已停止) ---
                # if hasattr(self, "diagnostics"):
                #     self.diagnostics.log_weights(step=self._train_step_count)
                
                self._train_step_count += 1

                # 记录 Loss
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                vicreg_losses.append(vicreg_loss.item())
                inv_dyn_losses.append(inv_dynamics_loss.item())

                # 显存清理：手动释放大张量引用，帮助垃圾回收
                # 注意：如果 batch_size 过大 (如 4096)，observations 可能会占用极高显存 (约 10GB+)
                # 建议将 batch_size 设置为 256 或 512，并通过 n_epochs 调节更新强度
                del observations, next_observations, features, next_features, latent_pi, latent_vf, distribution
                del policy_loss, value_loss, vicreg_loss, inv_dynamics_loss, loss, rollout_data
                
                # 每 10 个 Batch 尝试清理一次显存碎片 (可选)
                if current_iter % 10 == 0:
                    torch.cuda.empty_cache()

        # --- 反向传播完成后：恢复 SceneManager 采集 ---
        try:
            # 尝试通过 get_attr 获取 VecEnv 中的 scene_manager
            scene_managers = self.env.get_attr("scene_manager")
            if scene_managers and scene_managers[0] is not None:
                scene_managers[0].setup()
                print("\n\033[92m[INFO] 已恢复 SceneManager 采集预览\033[0m")
        except (AttributeError, IndexError):
            # 如果失败，尝试直接访问 (针对非 VecEnv)
            if hasattr(self.env, "scene_manager"):
                self.env.scene_manager.setup()
                print("\n\033[92m[INFO] 已恢复 SceneManager 采集预览\033[0m")

        print("\n\033[92m[TRAIN] 反向传播训练完成。\033[0m")
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/vicreg_loss", np.mean(vicreg_losses))
        self.logger.record("train/inv_dyn_loss", np.mean(inv_dyn_losses))
        self.logger.record("train/explained_variance", explained_var)
