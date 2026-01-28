"""
模块用途：视频帧与经验的回放缓冲（环形队列），支持帧堆叠与批量采样。

包含：
- 函数：sample_n_unique(sampling_f, n)
- 类：ReplayBuffer（store_frame/store_effect/get_latest_observation/sample 等）

边界：
- 负责：存储与返回训练需要的 (obs, act, rew, next_obs, done)
- 不负责：模型训练逻辑与奖励计算
"""

import numpy as np
import random

def sample_n_unique(sampling_f, n):
    """
    从采样源中不重复地抽取 n 个样本。
    Args:
        sampling_f: 一个可调用对象，每次调用返回一个样本。
        n: 需要抽取的样本数量。
    Returns:
        包含 n 个不重复样本的列表。
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """
        初始化经验回放缓冲区。
        Args:
            size: 回放缓冲区的最大容量。
            frame_history_len: 帧历史长度，即每个状态包含的连续帧数。
        """
        super().__init__()

        self.size = size
        self.frame_history_len = frame_history_len

        self.video_next_idx = 0
        self.video_num_in_buffer = 0

        self.experience_next_idx = 0 # New: for experience buffer
        self.experience_num_in_buffer = 0 # New: for experience buffer

        self.video_obs = None
        self.obs = None
        self.action = np.empty([self.size], dtype=np.int32)
        self.reward = np.empty([self.size], dtype=np.float32)
        self.done = np.empty([self.size], dtype=np.bool_)
        self.priority = np.ones([self.size], dtype=np.float32)
        self.max_priority = 1.0

    def store_frame(self, frame):
        """
        存储单个帧到视频缓冲区。
        Args:
            frame: 要存储的帧数据。
        Returns:
            当前帧在视频缓冲区中的索引。
        """
        # 判断输入是否为图像帧：当 shape 长度 > 1 时，通常为 HWC（高、宽、通道）或 HW（灰度）
        # 训练/堆叠阶段普遍采用通道优先的 CHW 布局，便于向量化与与深度学习框架适配
        if len(frame.shape) > 1:
            # 将图像帧维度从 HWC 转换为 CHW（通道、高、宽）。
            # 注意：若为灰度图（HW），transpose 不会触发，这里默认输入为彩色 RGB/BGR（HWC）。
            frame = frame.transpose(2, 0, 1)

        # 首次写入时初始化视频环形缓冲区：形状为 [K, C, H, W]
        # - K = frame_history_len（用于堆叠的连续帧数量）
        # - C/H/W 来源于单帧的 CHW 形状
        # 采用 uint8 节省内存并与图像像素范围 [0, 255] 对齐
        if self.video_obs is None:
            self.video_obs = np.empty([self.frame_history_len] + list(frame.shape), dtype=np.uint8)

        # 将当前帧写入环形缓冲区的当前位置（video_next_idx）
        # 不做拷贝之外的处理，保持原始像素数据以便后续堆叠/可视化
        self.video_obs[self.video_next_idx] = frame
        # print(f'video_obs shape: {self.video_obs.shape}')
        # 返回写入位置索引（供调用方在同一时间点生成堆叠状态）
        ret = self.video_next_idx

        # 前移写入指针并取模，实现环形覆盖：新帧写入将覆盖最旧帧
        self.video_next_idx = (self.video_next_idx + 1) % self.frame_history_len

        # 统计缓冲区已填充的有效帧数量：从 0 增长到 K 后保持不变
        # 该值用于判断堆叠是否具备足够的上下文帧
        self.video_num_in_buffer = min(self.frame_history_len, self.video_num_in_buffer + 1)

        return ret


    def get_latest_frame(self):
        """
        获取视频缓冲区的最近单帧观察。
        Returns:
            最近的单帧观察。
        """
        assert self.video_num_in_buffer > 0
        return self.video_obs[(self.video_next_idx - 1) % self.frame_history_len]

    def get_latest_observation(self, num_frames):
        """
        获取视频缓冲区的最近 num_frames 帧（时间顺序从旧到新，最后一帧为最新）。
        Args:
            num_frames: 需要堆叠的连续最近帧数量（上限为 frame_history_len）。
        Returns:
            (C*num_frames, H, W) 的堆叠张量，按时间顺序拼接（最旧在前，最新在后）。
        """
        # 至少已有一帧被写入，否则无法返回“最近帧”
        assert self.video_num_in_buffer > 0

        # 规范化请求帧数：下限 1，上限为环形缓冲容量 frame_history_len
        k = int(num_frames)
        if k < 1:
            k = 1
        if k > self.frame_history_len:
            k = self.frame_history_len

        # 写入指针 video_next_idx 指向“下一次写入位置”，因此最新帧在 (video_next_idx - 1)
        # 我们选择半开区间 [start_idx, end_idx) 来获取 k 帧，保证顺序为：从最旧到最新
        end_idx = self.video_next_idx
        start_idx = end_idx - k

        # 缓冲尚未填满且出现负起点时，将起点截到 0，后续用 missing_context 补零保证长度为 k
        if start_idx < 0 and self.video_num_in_buffer != self.frame_history_len:
            start_idx = 0

        # 计算实际可用帧与需求帧的差额；>0 时需要在序列前补零帧，保持时间顺序与长度一致
        missing_context = k - (end_idx - start_idx)

        if start_idx < 0 or missing_context > 0:
            # 补齐缺失的早期上下文：零帧在最前，随后依次追加真实帧（时间顺序保持：旧→新）
            frames = [np.zeros_like(self.video_obs[0]) for _ in range(missing_context)]
            for i in range(start_idx, end_idx):
                # 取模实现环形读取，确保 [start_idx, end_idx) 之间的帧按时间顺序取出
                frames.append(self.video_obs[i % self.frame_history_len])
            # 直接在通道维（第一维）拼接，得到 (C*k, H, W)
            return np.concatenate(frames, 0)
        else:
            # 当缓冲已填满或足够时，直接切片得到形状为 (k, C, H, W)
            # 通过 reshape(-1, H, W) 将时间维与通道维合并为通道堆叠，顺序依旧为旧→新
            img_h, img_w = self.video_obs.shape[2], self.video_obs.shape[3]
            return self.video_obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_latest_observation(self, current_obs_stacked):
        """
        将最近的堆叠观察写入经验缓冲区的当前索引（不推进索引）。
        """
        if self.obs is None:
            self.obs = np.empty([self.size] + list(current_obs_stacked.shape), dtype=np.uint8)
        self.obs[self.experience_next_idx] = current_obs_stacked

    def store_effect(self, action, reward, done):
        """
        将动作、奖励、完成标志写入经验缓冲区的当前索引，并推进索引。
        """
        self.action[self.experience_next_idx] = action
        self.reward[self.experience_next_idx] = reward
        self.done[self.experience_next_idx] = done
        self.priority[self.experience_next_idx] = self.max_priority

        self.experience_next_idx = (self.experience_next_idx + 1) % self.size
        self.experience_num_in_buffer = min(self.size, self.experience_num_in_buffer + 1)


    def can_sample(self, batch_size):
        """
        检查经验缓冲区是否可以采样指定批次大小的经验。
        Args:
            batch_size: 批次大小。
        Returns:
            如果可以采样，则为 True，否则为 False。
        """
        return batch_size + 1 <= self.experience_num_in_buffer # Modified

    def can_sample_n_step(self, batch_size, n_step):
        """
        检查是否可进行 N 步回报采样：需要至少 (n_step+1) 条连续经验，且起点数量足够。
        """
        if self.experience_num_in_buffer < (n_step + 1):
            return False
        max_start = self.experience_num_in_buffer - (n_step + 1)
        # 可选起点数量 = max_start + 1，需满足批次大小不超过该数量
        return batch_size <= (max_start + 1)

    def sample(self, batch_size):
        """
        从经验缓冲区中采样一个批次的经验。
        Args:
            batch_size: 批次大小。
        Returns:
            包含观察、动作、奖励、下一个观察和完成掩码的元组。
        """
        assert self.can_sample(batch_size)
        # 随机采样不重复的索引
        idxes = sample_n_unique(lambda: random.randint(0, self.experience_num_in_buffer - 2), batch_size) # Modified
        return self._encode_sample(idxes)

    def sample_per(self, batch_size, alpha=0.6, beta=0.4, eps=1e-6):
        assert self.can_sample(batch_size)
        valid_n = self.experience_num_in_buffer
        p = self.priority[:valid_n].copy()
        p = np.clip(p, eps, None)
        p = p ** float(alpha)
        p = p / float(p.sum())
        idxes = np.random.choice(valid_n - 1, size=batch_size, replace=False, p=p[:-1])
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self._encode_sample(idxes)
        w = (valid_n * p[idxes]) ** (-float(beta))
        w = w / float(w.max())
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, np.array(idxes, dtype=np.int32), w.astype(np.float32)

    def update_priorities(self, idxes, td_errors, eps=1e-6):
        td = np.abs(td_errors).astype(np.float32) + float(eps)
        self.priority[idxes] = td
        self.max_priority = float(max(self.max_priority, float(td.max())))

    def _encode_sample(self, idxes):
        """
        根据给定的索引编码采样经验。
        Args:
            idxes: 采样到的索引列表。
        Returns:
            包含观察、动作、奖励、下一个观察和完成掩码的元组。
        """
        obs_batch = np.concatenate([self.obs[idx][None] for idx in idxes], 0) # Modified
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self.obs[(idx + 1) % self.size][None] for idx in idxes], 0) # Modified
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample_n_step(self, batch_size, n_step, gamma):
        """
        采样N步回报样本：返回 (obs, act, R_n, next_obs_n, done_n)
        - R_n = sum_{k=0..T-1} gamma^k * r_{t+k}，T<=n_step，遇到done提前停止
        - next_obs_n = obs[t+T]
        - done_n = 1.0 如果在T步内出现终止，否则0.0
        """
        assert n_step >= 1
        if self.experience_num_in_buffer < (n_step + 1):
            raise AssertionError("insufficient buffer for n-step sampling")
        max_start = self.experience_num_in_buffer - (n_step + 1)
        idxes = sample_n_unique(lambda: random.randint(0, max_start), batch_size)
        obs_batch = np.concatenate([self.obs[idx][None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        Rn = np.empty([batch_size], dtype=np.float32)
        next_obs_idx = np.empty([batch_size], dtype=np.int32)
        done_mask = np.empty([batch_size], dtype=np.float32)
        steps_used = np.empty([batch_size], dtype=np.int32)
        for bi, t in enumerate(idxes):
            acc = 0.0
            T = 0
            done_flag = 0.0
            for k in range(n_step):
                idx_k = (t + k) % self.size
                acc += (gamma ** k) * float(self.reward[idx_k])
                T = k + 1
                if bool(self.done[idx_k]):
                    done_flag = 1.0
                    break
            Rn[bi] = acc
            next_obs_idx[bi] = (t + T) % self.size
            done_mask[bi] = done_flag
            steps_used[bi] = T
        next_obs_batch = np.concatenate([self.obs[idx][None] for idx in next_obs_idx.tolist()], 0)
        return obs_batch, act_batch, Rn, next_obs_batch, done_mask, steps_used

    def sample_n_step_per(self, batch_size, n_step, gamma, alpha=0.6, beta=0.4, eps=1e-6):
        assert n_step >= 1
        if self.experience_num_in_buffer < (n_step + 1):
            raise AssertionError("insufficient buffer for n-step sampling")
        max_start = self.experience_num_in_buffer - (n_step + 1)
        starts = np.arange(max_start + 1, dtype=np.int32)
        p = self.priority[:max_start + 1].copy()
        p = np.clip(p, eps, None)
        p = p ** float(alpha)
        p = p / float(p.sum())
        idxes = np.random.choice(starts, size=batch_size, replace=False, p=p)
        obs_batch = np.concatenate([self.obs[idx][None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        Rn = np.empty([batch_size], dtype=np.float32)
        next_obs_idx = np.empty([batch_size], dtype=np.int32)
        done_mask = np.empty([batch_size], dtype=np.float32)
        steps_used = np.empty([batch_size], dtype=np.int32)
        for bi, t in enumerate(idxes.tolist()):
            acc = 0.0
            T = 0
            done_flag = 0.0
            for k in range(n_step):
                idx_k = (t + k) % self.size
                acc += (gamma ** k) * float(self.reward[idx_k])
                T = k + 1
                if bool(self.done[idx_k]):
                    done_flag = 1.0
                    break
            Rn[bi] = acc
            next_obs_idx[bi] = (t + T) % self.size
            done_mask[bi] = done_flag
            steps_used[bi] = T
        next_obs_batch = np.concatenate([self.obs[idx][None] for idx in next_obs_idx.tolist()], 0)
        w = ((max_start + 1) * p[idxes]) ** (-float(beta))
        w = w / float(w.max())
        return obs_batch, act_batch, Rn, next_obs_batch, done_mask, steps_used, np.array(idxes, dtype=np.int32), w.astype(np.float32)

