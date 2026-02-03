import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple, Generator, Optional, Union, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutBufferSamples

class SekiroRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_observations: torch.Tensor # 新增：用于逆动力学损失

class SekiroRolloutBuffer(RolloutBuffer):
    """
    显存优化型 RolloutBuffer：
    1. 内部只存储单帧图像，采样时动态堆叠。
    2. 支持存储 next_observations 用于辅助任务。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 显存优化：修改 observations 的形状为单帧 (3, H, W)
        # 注意：基类 __init__ 已经分配了空间，我们需要重新分配
        self.obs_shape = (3, self.obs_shape[1], self.obs_shape[2])
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.uint8)
        # 增加 next_observations 存储
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.uint8)

    def add(self, obs, action, reward, episode_start, value, log_prob, next_obs=None):
        """扩展 add 方法，支持传入 next_obs"""
        if next_obs is not None:
            # 同样需要处理 next_obs 的通道切片
            if next_obs.ndim == 3 and next_obs.shape[0] > 3:
                next_obs = next_obs[-3:]
            elif next_obs.ndim == 4 and next_obs.shape[1] > 3:
                next_obs = next_obs[:, -3:]
            self.next_observations[self.pos] = np.array(next_obs).copy()
        
        # 确保只存单帧（针对 VecEnv 自动堆叠的情况进行降维）
        # 如果维度是 (C, H, W) 且 C > 3 (单环境堆叠)
        if obs.ndim == 3 and obs.shape[0] > 3:
            obs = obs[-3:]
        # 如果维度是 (N, C, H, W) 且 C > 3 (多环境堆叠)
        elif obs.ndim == 4 and obs.shape[1] > 3:
            obs = obs[:, -3:]
            
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[SekiroRolloutBufferSamples, None, None]:
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        for start_idx in range(0, self.buffer_size * self.n_envs, batch_size):
            yield self._get_samples(indices[start_idx : start_idx + batch_size])

    def _get_samples(self, batch_indices: np.ndarray, env: Optional[GymEnv] = None) -> SekiroRolloutBufferSamples:
        """动态构建堆叠帧"""
        n_stack, skip = 4, 3
        
        # 提取基础数据
        obs_batch = []
        next_obs_batch = []
        
        for idx in batch_indices:
            step_idx = idx // self.n_envs
            env_idx = idx % self.n_envs
            
            # 1. 构建当前观测的堆叠帧 (s_t)
            stacked_obs = self._stack_frames(step_idx, env_idx, self.observations, n_stack, skip)
            obs_batch.append(stacked_obs)
            
            # 2. 构建下一帧观测的堆叠帧 (s_{t+1})
            # 逆动力学需要 z_t 和 z_{t+1}
            # z_{t+1} 的堆叠帧由 [s_{t-6}, s_{t-3}, s_t, s_{t+1}] 组成
            # 这里简化处理：直接使用 next_observations 替换掉堆叠中的最后一帧
            stacked_next_obs = self._stack_frames(step_idx, env_idx, self.observations, n_stack - 1, skip)
            # 拼接最新的下一帧
            current_next_obs = self.next_observations[step_idx, env_idx]
            stacked_next_obs = np.concatenate([stacked_next_obs, current_next_obs], axis=0)
            next_obs_batch.append(stacked_next_obs)

        # 转换为 Tensor
        data = (
            self.to_torch(np.stack(obs_batch)),
            self.to_torch(self.actions[batch_indices // self.n_envs, batch_indices % self.n_envs]),
            self.to_torch(self.values[batch_indices // self.n_envs, batch_indices % self.n_envs].flatten()),
            self.to_torch(self.log_probs[batch_indices // self.n_envs, batch_indices % self.n_envs].flatten()),
            self.to_torch(self.advantages[batch_indices // self.n_envs, batch_indices % self.n_envs].flatten()),
            self.to_torch(self.returns[batch_indices // self.n_envs, batch_indices % self.n_envs].flatten()),
            self.to_torch(np.stack(next_obs_batch))
        )
        return SekiroRolloutBufferSamples(*data)

    def _stack_frames(self, step_idx, env_idx, buffer_array, n_stack, skip):
        """辅助方法：从缓冲区提取单帧并拼接"""
        frames = []
        for i in range(n_stack):
            # 目标索引：当前步减去跳帧偏移
            target_idx = step_idx - (n_stack - 1 - i) * skip
            # 处理边界：如果索引小于 0，则重复第一帧
            if target_idx < 0:
                target_idx = 0
            frames.append(buffer_array[target_idx, env_idx])
        return np.concatenate(frames, axis=0)

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
            
            actions = actions.cpu().numpy()
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

        for epoch in range(self.n_epochs):
            # 每一轮 epoch 都打乱一次 buffer 数据
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()
                
                # --- 0. 显式归一化与数值保护 ---
                observations = rollout_data.observations
                if observations.dtype == torch.uint8:
                    observations = observations.float() / 255.0
                
                next_observations = rollout_data.next_observations
                if next_observations.dtype == torch.uint8:
                    next_observations = next_observations.float() / 255.0
                
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
                # 强制每个特征维度的标准差保持在一定水平，防止坍缩
                std_features = torch.sqrt(features.var(dim=0) + 1e-04)
                vicreg_loss = torch.mean(F.relu(1.0 - std_features))

                # --- 3. 辅助任务：Inverse Dynamics Loss ---
                # 预测动作：根据 z_t 和 z_{t+1} 预测 a_t
                next_features = self.policy.features_extractor(next_observations)
                pred_actions = self.inv_dyn_head(torch.cat([features, next_features], dim=1))
                inv_dynamics_loss = F.cross_entropy(pred_actions, actions)
                
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

                # 优化步骤
                self.policy.optimizer.zero_grad()
                loss.backward()
                
                # --- 诊断：通过绑定在模型上的 diagnostics 直接记录 ---
                if hasattr(self, "diagnostics"):
                    self.diagnostics.log_gradients(step=self._train_step_count)
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                # --- 诊断：记录更新后的权重 ---
                if hasattr(self, "diagnostics"):
                    self.diagnostics.log_weights(step=self._train_step_count)
                
                self._train_step_count += 1

                # 记录 Loss
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                vicreg_losses.append(vicreg_loss.item())
                inv_dyn_losses.append(inv_dynamics_loss.item())

                # 显存清理：手动释放大张量引用，帮助垃圾回收
                del observations, next_observations, features, next_features, latent_pi, latent_vf, distribution
                del policy_loss, value_loss, vicreg_loss, inv_dynamics_loss, loss

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/vicreg_loss", np.mean(vicreg_losses))
        self.logger.record("train/inv_dyn_loss", np.mean(inv_dyn_losses))
        self.logger.record("train/explained_variance", explained_var)
