import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer, BaseBuffer
from stable_baselines3.common.type_aliases import GymEnv, RolloutBufferSamples
from typing import NamedTuple, Generator, Optional, Union

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
    2. 支持存储 next_observations 用于辅助任务（如逆动力学）。
    3. 设计意义：相比原生的 RolloutBuffer + VecFrameStack，
       内存占用降低约 75%（4帧堆叠时），且支持跨步跳帧堆叠。
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: GymEnv,
        action_space: GymEnv,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        # 直接调用 BaseBuffer 的初始化，避开 RolloutBuffer 的 reset 逻辑
        BaseBuffer.__init__(self, buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        
        # 显存优化：计算单帧形状
        self.single_frame_shape = (3, self.obs_shape[1], self.obs_shape[2])
        
        # 提前定义好所有数组，防止 reset 逻辑混乱
        self.observations = None
        self.actions = None
        self.rewards = None
        self.returns = None
        self.episode_starts = None
        self.values = None
        self.log_probs = None
        self.advantages = None
        
        self.reset()

    def reset(self) -> None:
        # 1. 初始化显存优化后的 observations (buffer_size + 1)
        self.observations = np.zeros((self.buffer_size + 1, self.n_envs, *self.single_frame_shape), dtype=np.uint8)
        
        # 2. 初始化其他标准 Rollout 数组
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self.terminal_obs = {}
        self.generator_ready = False
        
        # 3. 调用 BaseBuffer.reset (它只做 pos = 0 和 full = False)
        # 注意：千万不要调用 super().reset()，因为它会调用 RolloutBuffer.reset() 再次覆盖 observations
        BaseBuffer.reset(self)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        next_obs: Optional[np.ndarray] = None,
    ) -> None:
        """扩展 add 方法，实现精妙的 next_obs 存储"""
        def _get_single_frame(x):
            if x.ndim == 3 and x.shape[0] > 3: return x[-3:]
            elif x.ndim == 4 and x.shape[1] > 3: return x[:, -3:]
            return x

        processed_obs = _get_single_frame(obs)
        
        # 处理 next_obs 的逻辑
        if next_obs is not None:
            processed_next = _get_single_frame(next_obs)
            # 当 pos == buffer_size - 1 时，它会存入那个额外的 +1 槽位
            self.observations[self.pos + 1] = np.array(processed_next).copy()

        super().add(processed_obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[SekiroRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling"
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        # 对齐 SB3 哲学：提前 swap_and_flatten 非观测数据以提高采样速度
        if not self.generator_ready:
            for tensor in ["actions", "values", "log_probs", "advantages", "returns", "episode_starts"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        start_idx = 0
        total_size = self.buffer_size * self.n_envs
        batch_size = batch_size if batch_size is not None else total_size
        
        while start_idx < total_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_indices: np.ndarray, env: Optional[GymEnv] = None) -> SekiroRolloutBufferSamples:
        """向量化构建堆叠帧，利用偏移获取 next_obs"""
        n_stack, skip = 4, 3
        batch_size = len(batch_indices)
        
        step_indices = batch_indices % self.buffer_size
        env_indices = batch_indices // self.buffer_size
        
        def _vectorized_stack(buffer_array, steps, envs):
            stack_offsets = np.arange(-(n_stack - 1) * skip, 1, skip)
            all_step_indices = steps[:, None] + stack_offsets[None, :]
            all_step_indices = np.clip(all_step_indices, 0, self.buffer_size) # 注意边界现在是 buffer_size
            samples = buffer_array[all_step_indices, envs[:, None]]
            return samples.reshape(batch_size, -1, *self.single_frame_shape[1:])

        obs_stack = _vectorized_stack(self.observations, step_indices, env_indices)
        
        # 精妙之处：next_obs 的堆叠只需要将索引整体偏移 +1
        # 因为我们有 buffer_size + 1 的空间，所以 step_indices + 1 永远合法
        next_obs_stack = _vectorized_stack(self.observations, step_indices + 1, env_indices)

        # 转换为 Tensor 并归一化
        obs_tensor = self.to_torch(obs_stack).float() / 255.0
        next_obs_tensor = self.to_torch(next_obs_stack).float() / 255.0
        
        data = (
            obs_tensor,
            self.to_torch(self.actions[batch_indices]),
            self.to_torch(self.values[batch_indices]),
            self.to_torch(self.log_probs[batch_indices]),
            self.to_torch(self.advantages[batch_indices]),
            self.to_torch(self.returns[batch_indices]),
            next_obs_tensor
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
