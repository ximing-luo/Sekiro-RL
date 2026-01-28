"""
模块用途：提供兴奋阈值与奖励阈值的机制，并支持事件分类函数。
具体功能：
- 每个动作有一个兴奋阈值，直接对模型输出的q值进行修改，经常被选择的动作阈值提供，不再容易被选择。
- 每个动作的每个事件有一个奖励阈值，一个动作触发同一事件的次数越多，这个事件给的奖励就越少。

包含：
- 类：DesireThreshold、RewardThreshold
- 函数：classify_event(env, action, events, prev_...)

边界：
- 负责：阈值状态维护与应用计算、事件分类规则
- 不负责：环境采集、训练循环、可视化与日志写入
"""

import torch
import json
import time
import os
from collections import deque
from typing import List, Dict, Any, Optional


class DesireThreshold:
    """维护每动作的动态兴奋阈值，结合 Dueling 映射、无动作专属与强效降温机制进行动作选择与阈值更新。

    外部核心参数：
    - num_actions：动作数量
    - initial_threshold：初始阈值（未提供 per-action 初始值时使用）
    - slow_revert_target：自动缓慢回落的目标阈值
    - window_seconds：近窗统计秒数
    - recent_reward_window：奖励历史窗口长度（默认 300）
    - initial_thresholds：每动作初始阈值列表/张量
    - no_op_consecutive_limit：连续无动作触发整体降阈的阈值
    - no_op_strong_drop_total：强效降温总降幅
    - spike_steps：冷却持续步数
    - spike_amount：强烈抬高幅度

    初始化步骤：
    1) 基本属性与初始阈值设置；
    2) 阈值表初始化（行0：量化阈值视图；行1：近窗选择次数）；并缓存初始阈值向量
    3) 数值边界与奖励抑制配置；
    4) 状态记录与事件队列/奖励历史；
    5) 无动作基础与计数器、自动回落计时器；
    6) Spike/冷却、强效降温与饥饿降温状态初始化；
    7) 初次量化阈值并建立 raw→量化视图同步（thresholds_raw → thresholds[0,:]）。

    重要实现约定：
    - thresholds_raw 为真实来源（原始连续值），所有阈值更新必须写入 thresholds_raw，再统一量化到 thresholds[0,:]；
    - thresholds[0,:] 仅作为量化后的只读视图，避免直接写入后被量化过程覆盖导致更新丢失。
    """

    def __init__(
        self,
        # 基础与初始化
        num_actions,  # 动作数量
        initial_threshold: float = -2.0,  # 初始阈值
        initial_thresholds: Optional[List[float]] | Optional[torch.Tensor] = None,  # 每动作初始阈值
        window_seconds: float = 10.0,  # 近窗统计秒数
        recent_reward_window: int = 15 * 5,  # 奖励历史窗口长度
        slow_revert_target: float = 1.0,  # 自动缓慢回落目标
        # Dueling 映射
        transform_alpha_ratio_state: float = 15.0,  # 状态值映射的 α
        transform_alpha_ratio_adv: float = 5.0,  # 优势值映射的 α
        # 强烈抬高/冷却
        spike_amount: float = 1.0,  # 强烈抬高幅度
        spike_steps: int = 100,  # 抬高幅度为1.0时需要的冷却持续步数
        spike_amounts: Optional[List[float]] | Optional[torch.Tensor] = None,  # 每动作抬高幅度列表
        # 无动作与强效降温
        no_op_action_idx: int = 0,  # 无动作索引
        no_op_consecutive_limit: int = 10,  # 连续无动作触发整体降阈阈值
        no_op_drop_value: float = 0.2,  # 整体降阈幅度（排除无动作）
        no_op_strong_drop_total: float = 0.5,  # 强效降温总降幅
        noop_raise_adjust: float = 0.2,  # 无动作抬升缩放
        noop_fall_adjust: float = 1.0,  # 无动作下降缩放
        # 饥饿强效降温（长期未选）
        starvation_window_seconds: Optional[float] = None,  # 若未提供则取 window_seconds
        starvation_strong_drop_total: float = 0.1,  # 单次饥饿降温总降幅（更温和）
        starvation_strong_duration_steps: int = 50,  # 饥饿降温持续步数（更平滑）
        starvation_retrigger_interval_seconds: float = 2.0,  # 再次触发的最小间隔
        # 设备
        device: Optional[torch.device] = None,  # 指定张量设备
    ):
        # 步骤 1：基本属性与初始阈值设置
        self.num_actions = int(num_actions)
        self.initial_threshold = float(initial_threshold)

        # 阈值表（形状为 (2, num_actions)）
        # 行 0：每个动作当前的兴奋阈值 a
        # 行 1：每个动作在最近 window_seconds 内被选中的次数（由事件时间戳统计得到）
        # 步骤 2：阈值表初始化（行0：阈值；行1：近窗选择次数）
        self.thresholds = torch.empty((2, self.num_actions), dtype=torch.float32)
        if initial_thresholds is not None:
            if isinstance(initial_thresholds, torch.Tensor):
                v = initial_thresholds.to(dtype=torch.float32).view(-1)
            else:
                v = torch.tensor(list(initial_thresholds), dtype=torch.float32).view(-1)
            if v.numel() != self.num_actions:
                raise ValueError("initial_thresholds length must equal num_actions")
            self.thresholds[0, :] = v
        else:
            self.thresholds[0, :] = self.initial_threshold
        self.thresholds[1, :] = 0.0
        # 缓存每动作初始阈值向量
        self.initial_threshold_vec = self.thresholds[0, :].clone()
        # 初始化 raw 阈值视图，供后续所有数值更新使用
        self.thresholds_raw = self.thresholds[0, :].clone()

        # 数值边界与系数
        self.threshold_min = -10.0  # 阈值下界，防止数值过低
        self.threshold_max = 10.0   # 阈值上界，防止数值过高
        self.window_seconds = float(window_seconds)  # 近窗统计秒数，用于计算选择次数
        self.short_term_reward_delta = 0.02  # 正奖励时的短期降阈幅度（更温和）
        self.inhibition_coeff = 0.2  # 过度选择时的抑制升阈系数
        self.low_count_threshold = 2  # 低选择次数阈值
        self.high_count_threshold = 5  # 高选择次数阈值

        # 记录最近一次选择的动作（用于奖励反馈）及事件时间戳队列
        # 步骤 4：状态记录与事件队列/奖励历史
        self.last_action = None  # 最近一次被选择的动作索引（反馈阶段定位目标动作）
        self._action_events = [deque() for _ in range(self.num_actions)]  # 每动作的事件时间戳队列，用于近窗选择次数统计
        self.recent_rewards = deque(maxlen=int(recent_reward_window))  # 近300步奖励历史，用于长期追求度
        self.life_value = 1.0  # 生命值归一化输入（[0,1]），结合长期平均奖励计算长期追求阈值
        self.long_term_min = -1.0  # 长期追求最小负值缩放因子，越负代表长期目标越严格
        self.auto_revert_interval = 0.5  # 自动回落触发的时间间隔（秒）
        self.auto_revert_step = 0.005  # 自动回落每步收敛的步进（阈值单位）
        # 步骤 5：自动回落计时器与无动作计数（无动作基础机制）
        self._last_auto_revert_ts = time.time()  # 上次自动回落触发的时间戳
        self._noop_count = 0  # 连续选择无动作的计数器（用于触发整体与强效降温）
        self.no_op_consecutive_limit = int(no_op_consecutive_limit)  # 连续无动作达到该阈值时触发整体降阈
        self.no_op_drop_value = no_op_drop_value  # 整体降阈的幅度（对所有动作）
        self.no_op_action_idx = int(no_op_action_idx)
        self.noop_raise_adjust = float(noop_raise_adjust)
        self.noop_fall_adjust = float(noop_fall_adjust)
        # 无动作初始阈值至少为 1.0，以避免早期无意义的主动选择
        self.thresholds[0, self.no_op_action_idx] = torch.clamp(self.thresholds[0, self.no_op_action_idx], min=1.5)
        # 无动作选择计数与类型标记
        self._noop_active_count = 0  # 连续主动选择无动作的计数
        self._noop_passive_count = 0  # 连续被动选择无动作的计数（无候选时）
        self._other_consecutive_count = 0  # 连续选择非无动作的计数
        self._last_noop_selected_active = False  # 最近一次选择无动作是否为主动选择
        self.select_raise_coeff = 0.05  # 历史兼容参数（强烈抬高已替代）
        self.slow_revert_target = float(slow_revert_target)  # 自动回落的收敛目标，而非初始阈值
        # 模块：Q值映射（Dueling）参数
        # 变换公式与调参说明：
        # - 公式：x_t = sign(x) * (|x| / (|x| + alpha)) * S
        # - S_state/S_adv = scale_base * transform_scale_ratio_*：增大 ratio 提高映射尺度，放大对应分量；
        # - alpha_state/alpha_adv = transform_alpha_ratio_*：增大 α ，用于压制高 Q值兴奋
        # - 调参建议：
        #   · 提高 transform_scale_ratio_state 可增强“状态值”权重（更依赖全局基线）；
        #   · 提高 transform_scale_ratio_adv 可增强“优势值”权重（更强调动作差异）；
        #   · 映射权重为1时，最大值为阈值的一半，两者和大于2会导致映射值超出阈值范围
        #   · 提高 transform_alpha_ratio_state/adv 会更强地将极大值分量压向 0（防止极端值让模型进入死循环）；
        #   · 两者配合用于均衡“基线 vs 动作差异”的影响力，达到稳定选择偏好。
        self.transform_scale_ratio_state = 0.6  # 状态值映射的尺度比例
        self.transform_scale_ratio_adv = 1.4    # 优势值映射的尺度比例  
        self.transform_alpha_ratio_state = transform_alpha_ratio_state  # 状态值映射的 α 这个值基本可以设为期望正确状态的完美状态值大小
        self.transform_alpha_ratio_adv = transform_alpha_ratio_adv  # 优势值映射的 α 这个值基本可以设为期望正确动作的完美优势值大小
        # 模块：无动作强效降温参数
        # 逻辑：连续/主动选择无动作触发强效降温，按步均幅下降，作用于Top-K最高阈值动作
        self.proactive_no_op_limit = max(1, int(self.no_op_consecutive_limit / 5))  # 主动触发阈值（较小）
        self.no_op_strong_limit = int(self.no_op_consecutive_limit)  # 连续触发阈值（等同连续无动作阈值）
        self.noop_passive_strong_threshold = 5  # 被动选择无动作触发强效降阈的次数阈值（内部常量）
        self.no_op_strong_duration_steps = 180  # 强效降温持续步数
        self.no_op_strong_drop_total = float(no_op_strong_drop_total)  # 强效降温总降幅（外部核心参数）
        self.no_op_top_k = 3  # 作用于最高阈值的前K个动作
        # 步骤 6：强烈抬高（spike）与冷却配置
        self.spike_steps = int(spike_steps)
        if spike_amounts is not None:
            if isinstance(spike_amounts, torch.Tensor):
                v = spike_amounts.to(dtype=torch.float32).view(-1)
            else:
                v = torch.tensor(list(spike_amounts), dtype=torch.float32).view(-1)
            if int(v.numel()) != int(self.num_actions):
                raise ValueError("spike_amounts length must equal num_actions")
            self.spike_amount_vec = v.clone()
        else:
            self.spike_amount_vec = torch.full((self.num_actions,), float(spike_amount), dtype=torch.float32)
        self.spike_decay_step_vec = torch.full((self.num_actions,), float(1.0 / max(1, float(self.spike_steps))), dtype=torch.float32)
        self.cooldown_remaining = torch.zeros((self.num_actions,), dtype=torch.int32)  # 每动作冷却剩余步数计数器，用于快速回落推进
        # 无动作强效降温配置与状态
        self.strong_cooling_remaining = torch.zeros((self.num_actions,), dtype=torch.int32)  # 每动作强效降温剩余步数计数器
        self.strong_cooling_step = torch.full((self.num_actions,), float(self.no_op_strong_drop_total / max(1, self.no_op_strong_duration_steps)), dtype=torch.float32)  # 每步降温的降幅大小
        # 饥饿强效降温参数与状态（内部配置）
        # 饥饿强效降温参数：当某动作长期未被选中时，触发与无动作强效降温类似的“饥饿降温”
        # 窗口时长：与近窗统计秒数一致，用于判断“多久没被选”
        self.starvation_window_seconds = int(window_seconds) if starvation_window_seconds is None else int(starvation_window_seconds)
        # 饥饿降温：采用更温和的总降幅与更长持续步数，避免快速整体下滑
        self.starvation_strong_drop_total = float(starvation_strong_drop_total)
        self.starvation_strong_duration_steps = int(starvation_strong_duration_steps)
        self.starvation_retrigger_interval_seconds = float(starvation_retrigger_interval_seconds)
        # 每动作上次触发饥饿降温的时间戳，初始化为 0 表示从未触发
        self.starvation_last_trigger_ts = [0.0 for _ in range(self.num_actions)]
        # 步骤 7：初次量化阈值（从 raw 同步到量化视图）
        self._quantize_thresholds()

        if device is not None:
            self.device = device
            self.thresholds = self.thresholds.to(device)
            self.thresholds_raw = self.thresholds_raw.to(device)
            self.initial_threshold_vec = self.initial_threshold_vec.to(device)
            self.cooldown_remaining = self.cooldown_remaining.to(device)
            self.strong_cooling_remaining = self.strong_cooling_remaining.to(device)
            self.strong_cooling_step = self.strong_cooling_step.to(device)
            self.spike_amount_vec = self.spike_amount_vec.to(device)
            self.spike_decay_step_vec = self.spike_decay_step_vec.to(device)
        else:
            self.device = self.thresholds.device

    def apply(self, q_values: torch.Tensor) -> tuple[torch.Tensor, int]:
        """职责：根据当前阈值与 Dueling 映射修正 Q，进行动作选择并处理无动作与冷却语义。

        输入：
        - q_values: 形状为 `(1, num_actions)` 的张量

        实现步骤：
        1) 设备对齐与输入校验；
        2) Dueling 变换得到收缩后的 `Q_t`；
        3) 候选 gating 与动作选择（含无动作优先与无候选回退）；
        4) 更新事件窗口计数；
        5) 非无动作被选时强烈抬高并启动冷却；
        6) 记录最近被选动作；
        7) 返回 `(q_transformed, action_idx)`。

        输出：
        - `(q_transformed, action_idx)`：`q_transformed` 为 Dueling 收缩后的 `Q_t`，`action_idx` 为本步选中的动作索引。
        """
        # 步骤 1：设备对齐与输入校验
        if q_values.dim() != 2 or q_values.size(0) != 1:
            raise ValueError("q_values must be shape (1, num_actions)")

        if self.thresholds.device != q_values.device:
            raise ValueError("q_values device mismatch; construct DesireThreshold with matching device")

        # 内部函数：Dueling 变换
        # - 分解 Q 为状态值/优势值并分别进行符号保留的绝对值收缩到阈值尺度；
        # - 返回 `Q_t`（收缩后的变换值）；
        # - 在动作选择阶段，使用与量化阈值的比较与差值 `(Q_t - thresholds)` 进行候选 gating 与抽样权重计算。
        # Dueling 变换步骤：
        # 1) 读取当前行 Q 并求均值作为状态值；2) 得到优势值；3) 计算尺度基准与 S/α；
        # 4) 加入数值稳定项 eps；5) 分别对状态/优势进行符号保留的绝对值收缩；6) 组合得到 `Q_t`。
        def _transform_dueling(q_values_: torch.Tensor) -> torch.Tensor:
            # 步骤2：计算当前行Q的状态值与优势值
            q_row_ = q_values_[0, :]  # 当前批唯一行的Q（A维）
            q_mean_ = torch.mean(q_row_)  # 状态值：Q的均值
            state_val_ = q_mean_  # 显式命名，便于后续映射
            adv_ = q_row_ - q_mean_  # 优势值：逐动作的偏差
            # 记录原始分解数据（供前端可视化与日志）
            try:
                self.last_state_value = float(state_val_.item())
                self.last_adv_raw = adv_.detach().cpu().tolist() if adv_.device.type != 'meta' else []
            except Exception:
                self.last_state_value = float(state_val_.item())
                self.last_adv_raw = []
            # 步骤3：计算映射尺度与α
            scale_base_ = (self.threshold_max - self.threshold_min) / 4.0  # 尺度基准（默认5），对应阈值范围的一半
            S_state_ = torch.tensor(scale_base_ * self.transform_scale_ratio_state, dtype=torch.float32, device=q_values_.device)  # 状态映射尺度
            S_adv_ = torch.tensor(scale_base_ * self.transform_scale_ratio_adv, dtype=torch.float32, device=q_values_.device)  # 优势映射尺度
            alpha_state_ = self.transform_alpha_ratio_state # 状态 α
            alpha_adv_ = self.transform_alpha_ratio_adv # 优势 α
            # 步骤4：将状态值与优势值按映射尺度与α，进行符号保留的绝对值收缩
            eps_ = torch.tensor(1e-6, dtype=torch.float32, device=q_values_.device)  # 数值稳定项，避免分母为零
            # 状态值符号保留的绝对值收缩
            state_t_ = torch.sign(state_val_) * (torch.abs(state_val_) / (torch.abs(state_val_) + alpha_state_ + eps_)) * S_state_  
            # 优势值逐元素收缩（符号保留）
            adv_t_ = torch.sign(adv_) * (torch.abs(adv_) / (torch.abs(adv_) + alpha_adv_ + eps_)) * S_adv_  
            try:
                self.last_adv_shrink = adv_t_.detach().cpu().tolist()
            except Exception:
                self.last_adv_shrink = []
            # 步骤5：组合收缩后的状态值与优势值
            q_transformed_ = (state_t_ + adv_t_).unsqueeze(0)
            return q_transformed_
        # 步骤 2：Dueling 变换
        modified = _transform_dueling(q_values)

        # 内部函数：动作选择
        # - 无动作优先：原始 q(no_op) 最高则直接选择无动作（免抽样、豁免抬高）
        # - 无候选：选择无动作，累计并触发整体降阈/强效降温
        # - 有候选：按 Q' 进行加权抽样
        def _select_action(modified_: torch.Tensor, q_values_: torch.Tensor) -> int:
            pos_ = (modified_[0, :] > self.thresholds[0, :])
            had_candidates_ = bool(torch.any(pos_).item())
            if bool(pos_[int(self.no_op_action_idx)].item()):
                aidx = int(self.no_op_action_idx)
                self._noop_count += 1
                self._noop_active_count += 1
                self._noop_passive_count = 0
                self._other_consecutive_count = 0
                self._last_noop_selected_active = True
                if self._noop_count >= self.no_op_consecutive_limit:
                    mask = torch.ones_like(self.thresholds_raw, dtype=torch.bool)
                    mask[int(self.no_op_action_idx)] = False
                    dec = torch.full_like(self.thresholds_raw, float(self.no_op_drop_value))
                    self.thresholds_raw = torch.clamp(self.thresholds_raw - dec * mask.float(), min=self.threshold_min, max=self.threshold_max)
                    self._noop_count = 0
                    self._quantize_thresholds()
                if had_candidates_ and self._noop_count >= self.proactive_no_op_limit:
                    a_vals_ = self.thresholds[0, :]
                    max_idx_ = int(torch.argmax(a_vals_).item())
                    if max_idx_ != int(self.no_op_action_idx):
                        self.strong_cooling_remaining[max_idx_] = int(self.no_op_strong_duration_steps)
                        self._noop_count = 0
                return aidx
            q_noop_ = q_values_[0, self.no_op_action_idx]
            if (q_noop_ >= torch.max(q_values_[0, :])) and bool(pos_[int(self.no_op_action_idx)].item()):
                aidx = int(self.no_op_action_idx)
                self._noop_count += 1
                self._noop_active_count += 1
                self._noop_passive_count = 0
                self._other_consecutive_count = 0
                self._last_noop_selected_active = True
                if self._noop_count >= self.no_op_consecutive_limit:
                    mask = torch.ones_like(self.thresholds_raw, dtype=torch.bool)
                    mask[int(self.no_op_action_idx)] = False
                    dec = torch.full_like(self.thresholds_raw, float(self.no_op_drop_value))
                    self.thresholds_raw = torch.clamp(self.thresholds_raw - dec * mask.float(), min=self.threshold_min, max=self.threshold_max)
                    self._noop_count = 0
                    self._quantize_thresholds()
                if had_candidates_ and self._noop_count >= self.proactive_no_op_limit:
                    a_vals_ = self.thresholds[0, :]
                    max_idx_ = int(torch.argmax(a_vals_).item())
                    if max_idx_ != int(self.no_op_action_idx):
                        self.strong_cooling_remaining[max_idx_] = int(self.no_op_strong_duration_steps)
                        self._noop_count = 0
                return aidx
            if not had_candidates_:
                aidx = int(self.no_op_action_idx)
                self._noop_count += 1
                self._noop_passive_count += 1
                self._noop_active_count = 0
                self._other_consecutive_count = 0
                self._last_noop_selected_active = False
                if self._noop_count >= self.no_op_consecutive_limit:
                    # 整体降阈时排除无动作索引，避免被动选择导致无动作阈值下降
                    mask = torch.ones_like(self.thresholds_raw, dtype=torch.bool)
                    mask[int(self.no_op_action_idx)] = False
                    dec = torch.full_like(self.thresholds_raw, float(self.no_op_drop_value))
                    self.thresholds_raw = torch.clamp(self.thresholds_raw - dec * mask.float(), min=self.threshold_min, max=self.threshold_max)
                    a_vals_ = self.thresholds[0, :]
                    _, topk_idx_ = torch.topk(a_vals_, k=min(self.no_op_top_k, self.num_actions))
                    for idx in topk_idx_.tolist():
                        if int(idx) == int(self.no_op_action_idx):
                            continue
                        self.strong_cooling_remaining[int(idx)] = int(self.no_op_strong_duration_steps)
                    self._noop_count = 0
                    self._quantize_thresholds()
                return aidx
            weights_ = (modified_[0, :] - self.thresholds[0, :])[pos_]
            idxs_ = torch.arange(self.num_actions, device=modified_.device)[pos_]
            sel_ = torch.multinomial(weights_, num_samples=1, replacement=False)
            self._noop_count = 0
            self._other_consecutive_count += 1
            self._last_noop_selected_active = False
            self._noop_active_count = 0
            self._noop_passive_count = 0
            return int(idxs_[sel_].item())
        # 步骤 3：候选 gating 与动作选择
        action_idx = _select_action(modified, q_values)

        # 步骤 4：记录时间戳并更新窗口内选择次数
        now = time.time()
        ev = self._action_events[action_idx]
        ev.append(now)
        cutoff = now - self.window_seconds
        while ev and ev[0] < cutoff:
            ev.popleft()
        self.thresholds[1, action_idx] = float(len(ev))

        # 步骤 5：被选动作强烈抬高并启动冷却倒计时（无动作豁免）
        if action_idx != int(self.no_op_action_idx):
            target_high = self.initial_threshold_vec[action_idx] + self.spike_amount_vec[action_idx]
            self.thresholds_raw[action_idx] = torch.clamp(target_high, min=self.threshold_min, max=self.threshold_max)
            # 若此动作正处于冷却，丢弃之前剩余量并重新开始新一轮冷却
            self.cooldown_remaining[action_idx] = int(torch.ceil(self.spike_amount_vec[action_idx] * float(self.spike_steps)).item())
            self._quantize_thresholds()

        # 步骤 6：记录最近被选动作（用于奖励反馈）
        self.last_action = action_idx

        return modified, action_idx

    def feedback(self, reward) -> None:
        """职责：集中处理所有阈值更新（冷却/强效降温、奖励分支、自动缓慢回落与量化）。

        阶段步骤（逐步注释）：

        1) 冷却期快速下降：对处于冷却的动作按固定步长自然回落至基线（不强制复位）；
        1b) 强效降温：若存在无动作触发的强效降温期，则按步均幅下降并递减持续计数；
        2) 奖励统一化与历史：统一 reward 为 float，记录到近 recent_reward_window 步历史并计算 avg_recent；
        3) 正奖励分支：低计数降阈，高计数按溢出量与抑制系数升阈；
        4) 负奖励分支：低于长期追求阈值则升阈抑制重复错误，否则降阈鼓励探索；
        5) 自动缓慢回落：周期性将所有动作的阈值向 slow_revert_target 轻微靠近；
        6) 量化收敛：所有更新完成后统一量化到整数 [-10, 10]。

        数值流向与安全事项：
        - 所有更新均作用于 `thresholds_raw`（连续值来源），随后通过 `_quantize_thresholds()` 同步到量化视图 `thresholds[0,:]`；
        - 设备一致性：创建临时张量时使用 `device=self.thresholds_raw.device` 或直接写入标量，避免 CPU→GPU 写入不一致；
        - 边界控制：所有写入前后均以 `threshold_min/max` 限制，保持稳定范围。
        """
        # 入参与前置条件：至少有一次动作被选择
        if self.last_action is None:
            return
        # 内部函数：应用冷却回落（自然抵消，不复位）
        def _apply_cooldown():
            if self.cooldown_remaining is None:
                return
            for i in range(self.num_actions):
                if int(self.cooldown_remaining[i].item()) > 0:
                    # 读取 raw 值作为真实来源，按步回落至初始值
                    curr = float(self.thresholds_raw[i].item())
                    base = float(self.initial_threshold_vec[i].item())
                    if curr > base:
                        new_v = max(base, curr - float(self.spike_decay_step_vec[i].item()))
                    else:
                        new_v = base
                    # 写回 raw，再统一量化
                    self.thresholds_raw[i] = torch.clamp(torch.tensor(new_v, dtype=torch.float32, device=self.thresholds_raw.device), min=self.threshold_min, max=self.threshold_max)
                    self.cooldown_remaining[i] = int(self.cooldown_remaining[i].item()) - 1
            self._quantize_thresholds()

        # 内部函数：应用无动作强效降温期（逐步降幅）
        def _apply_strong_cooling():
            for i in range(self.num_actions):
                if i == int(self.no_op_action_idx):
                    continue
                if int(self.strong_cooling_remaining[i].item()) > 0:
                    # 强效降温直接作用于 raw 值
                    curr = float(self.thresholds_raw[i].item())
                    dec = float(self.strong_cooling_step[i].item())
                    self.thresholds_raw[i] = torch.clamp(torch.tensor(curr - dec, dtype=torch.float32, device=self.thresholds_raw.device), min=self.threshold_min, max=self.threshold_max)
                    self.strong_cooling_remaining[i] = int(self.strong_cooling_remaining[i].item()) - 1
            self._quantize_thresholds()

        # 步骤 1：冷却期快速下降（自然回落至基线）
        _apply_cooldown()
        # 步骤 1b：强效降温（由无动作/饥饿等机制触发）
        _apply_strong_cooling()

        # 步骤 2：统一奖励为浮点并记录到近窗历史（长度 recent_reward_window）  
        # 作用  
        # 将传入的 reward 统一为 float ，避免 Tensor 与 float 混用导致的类型问题.
        # 将本步奖励写入长度 recent_reward_window 的历史队列 recent_rewards，并计算其平均值 avg_recent 作为长期表现指标。
        # 后续用途
        # 在负奖励分支中，用 avg_recent 与 life_value 共同计算长期追求阈值 ltp = _compute_long_term_pursuit(life_value, avg_recent)。
        # 据此决定是升阈抑制重复错误，还是降阈鼓励探索。
        # life_value 在训练中按自身血量归一化后每步更新，与 avg_recent 形成长期评价信号。 

        # 内部函数：奖励统一化与近300步均值计算
        def _normalize_reward(rw):
            if isinstance(rw, torch.Tensor):
                rv = float(rw.item())
            else:
                rv = float(rw)
            self.recent_rewards.append(rv)
            avg_recent = sum(self.recent_rewards) / len(self.recent_rewards) if len(self.recent_rewards) > 0 else 0.0
            return rv, avg_recent
        reward_val, avg_recent = _normalize_reward(reward)

        # 步骤 3：正奖励分支
        # 内部函数：正/负奖励分支阈值更新
        # 奖励分支（核心机制）：决定被选动作的阈值是升还是降
        # 数值建议：
        # - short_term_reward_delta（默认 0.2）：若希望量化到整数后立即生效，可提高至 ≥1.0；否则用 0.2 形成渐进式影响
        # - inhibition_coeff（默认 0.05）：建议 0.02–0.10；选择次数溢出越大，升阈越强，抑制过度选择
        # - low/high_count_threshold（默认 10/10）：可按环境步频设置，如 low≈0.3×window_seconds, high≈0.7×window_seconds
        # - 量化提醒：阈值最终向下取整到整数；过小的增减需要累计多次才可见
        def _apply_reward_branch(reward_val_: float, avg_recent_: float) -> None:
            # 无动作：完全豁免标准奖励分支，改由专属机制处理
            if int(self.last_action) == int(self.no_op_action_idx):
                return
            # 正奖励：鼓励低频动作，抑制过度选择动作
            if reward_val_ > 0.0:
                c_ = int(self.thresholds[1, self.last_action].item())  # 近窗选择次数
                if c_ < self.low_count_threshold:
                    # 低频选择：降阈，降低选择门槛，鼓励继续选择
                    t_ = self.thresholds_raw[self.last_action] - self.short_term_reward_delta
                    self.thresholds_raw[self.last_action] = torch.clamp(t_, min=self.threshold_min, max=self.threshold_max)
                    self._quantize_thresholds()
                elif c_ > self.high_count_threshold:
                    # 过度选择：按溢出量升阈，提高门槛，打散过度偏好
                    overflow_ = c_ - self.high_count_threshold
                    t_ = self.thresholds_raw[self.last_action] + overflow_ * self.inhibition_coeff
                    self.thresholds_raw[self.last_action] = torch.clamp(t_, min=self.threshold_min, max=self.threshold_max)
                    self._quantize_thresholds()
            # 负奖励：若显著劣于长期追求，则升阈避免重复错误；否则降阈鼓励探索
            elif reward_val_ < 0.0:
                c_ = int(self.thresholds[1, self.last_action].item())  # 近窗选择次数
                ltp_ = self._compute_long_term_pursuit(self.life_value, avg_recent_)  # 长期追求阈值（由生命值与近窗均值决定）
                if reward_val_ < ltp_:
                    # 显著劣于长期目标：升阈避免重复错误（高频时强升）
                    if c_ > self.high_count_threshold:
                        overflow_ = c_ - self.high_count_threshold
                        t_ = self.thresholds_raw[self.last_action] + overflow_ * self.inhibition_coeff
                    else:
                        t_ = self.thresholds_raw[self.last_action] + self.inhibition_coeff
                    self.thresholds_raw[self.last_action] = torch.clamp(t_, min=self.threshold_min, max=self.threshold_max)
                    self._quantize_thresholds()
                else:
                    # 接近或好于长期目标：降阈鼓励探索与修复
                    t_ = self.thresholds_raw[self.last_action] - self.short_term_reward_delta
                    self.thresholds_raw[self.last_action] = torch.clamp(t_, min=self.threshold_min, max=self.threshold_max)
                    self._quantize_thresholds()
        _apply_reward_branch(reward_val, avg_recent)

        # 无动作阈值升降专属机制：主动选择抑制无限选择、在其他动作选择后适度回落、被动选择过多加强全局降阈
        def _apply_noop_branch(avg_recent_: float) -> None:
            idx = int(self.no_op_action_idx)
            # 主动选择无动作：按连续次数分级升阈，避免无限选择
            if self._last_noop_selected_active:
                L = max(1, int(self.proactive_no_op_limit))
                c = int(self._noop_active_count)
                if c >= L:
                    if c < int(1.4 * L):
                        base = 1.0
                    elif c < int(2.0 * L):
                        base = 1.4
                    else:
                        base = 2.0 + 0.2 * max(0, (c - int(2.0 * L)) / float(L))
                    avg_clamped = max(-1.0, min(1.0, avg_recent_))
                    factor = (1.0 - 0.5 * max(0.0, avg_clamped)) if avg_clamped > 0.0 else (1.0 + 0.5 * abs(avg_clamped))
                    delta = base * factor * float(self.noop_raise_adjust)
                    t = self.thresholds_raw[idx] + delta
                    self.thresholds_raw[idx] = torch.clamp(t, min=self.threshold_min, max=self.threshold_max)
                    self._quantize_thresholds()
            # 在其他动作被选择后：对无动作阈值小幅/加大回落，鼓励时机切换
            if self._other_consecutive_count >= 1 and self.thresholds_raw[idx] > 3:
                dec = (0.03 if self._other_consecutive_count <= 5 else 0.01) * float(self.noop_fall_adjust)
                t = self.thresholds_raw[idx] - dec
                base_min = max(1.5, float(self.initial_threshold_vec[idx].item()))
                t = max(base_min, float(t))
                self.thresholds_raw[idx] = torch.clamp(torch.tensor(t, dtype=torch.float32, device=self.thresholds_raw.device), min=self.threshold_min, max=self.threshold_max)
                self._quantize_thresholds()
            # 被动选择无动作过多：加强全局降阈，打散可能的局部锁死
            if int(self._noop_passive_count) > int(self.noop_passive_strong_threshold):
                a_vals = self.thresholds[0, :]
                _, topk_idx = torch.topk(a_vals, k=min(self.no_op_top_k, self.num_actions))
                step_big = float(self.no_op_strong_drop_total / max(1, self.no_op_strong_duration_steps)) * 2.0
                for j in topk_idx.tolist():
                    if int(j) == int(self.no_op_action_idx):
                        continue
                    self.strong_cooling_remaining[int(j)] = int(self.no_op_strong_duration_steps)
                    self.strong_cooling_step[int(j)] = float(step_big)
                self._noop_passive_count = 0
            # 每步自动向初始值归位 0.02
            curr_ = float(self.thresholds_raw[idx].item())
            base_ = float(self.initial_threshold_vec[idx].item())
            if curr_ > base_:
                t_ = max(base_, curr_ - 0.02)
            elif curr_ < base_:
                t_ = min(base_, curr_ + 0.02)
            else:
                t_ = base_
            self.thresholds_raw[idx] = torch.clamp(torch.tensor(t_, dtype=torch.float32, device=self.thresholds_raw.device), min=self.threshold_min, max=self.threshold_max)
            self._quantize_thresholds()

        _apply_noop_branch(avg_recent)

        # 饥饿强效降温：在饥饿窗口内未被选择的动作触发强效降温期（排除无动作）
        def _apply_starvation_cooling() -> None:
            now_s = time.time()
            cutoff = now_s - self.starvation_window_seconds
            for i in range(self.num_actions):
                if i == int(self.no_op_action_idx):
                    continue
                ev = self._action_events[i]
                while ev and ev[0] < cutoff:
                    ev.popleft()
                cnt = len(ev)
                if cnt == 0 and int(self.strong_cooling_remaining[i].item()) == 0:
                    last_ts = float(self.starvation_last_trigger_ts[i]) if self.starvation_last_trigger_ts[i] is not None else 0.0
                    if (now_s - last_ts) >= self.starvation_retrigger_interval_seconds:
                        self.strong_cooling_remaining[i] = int(self.starvation_strong_duration_steps)
                        self.strong_cooling_step[i] = float(self.starvation_strong_drop_total / max(1, self.starvation_strong_duration_steps))
                        self.starvation_last_trigger_ts[i] = now_s
        _apply_starvation_cooling()
                
        # 步骤 5：自动回落（周期性将所有动作阈值向各自初始值轻微靠近，维持数值稳定）
        # 内部函数：自动缓慢回落至 slow_revert_target
        def _auto_revert():
            now2_ = time.time()
            if now2_ - self._last_auto_revert_ts >= self.auto_revert_interval:
                # 自动回落应作用于 raw 值，再统一量化
                a_ = self.thresholds_raw
                diff_ = self.slow_revert_target - a_
                step_ = torch.where(torch.abs(diff_) < self.auto_revert_step, diff_, torch.sign(diff_) * self.auto_revert_step)
                if int(self.no_op_action_idx) < self.num_actions:
                    step_[int(self.no_op_action_idx)] = torch.tensor(0.0, dtype=torch.float32, device=self.thresholds_raw.device)
                self.thresholds_raw = torch.clamp(a_ + step_, min=self.threshold_min, max=self.threshold_max)
                self._quantize_thresholds()
                self._last_auto_revert_ts = now2_
                if int(self.no_op_action_idx) < self.num_actions:
                    i_ = int(self.no_op_action_idx)
                    v_ = float(self.thresholds_raw[i_].item())
                    self.thresholds_raw[i_] = torch.clamp(torch.tensor(max(1.5, v_), dtype=torch.float32), min=self.threshold_min, max=self.threshold_max)
                    self._quantize_thresholds()
        _auto_revert()

    def get_recent_action_counts(self, exclude_noop: bool = True) -> List[int]:
        """职责：返回近窗口内各动作的被选次数。

        输入：
        - `exclude_noop`: 是否将无动作索引的计数置零（默认 True）。

        实现步骤：
        - 清理各动作事件队列中过期时间戳；
        - 根据开关决定是否对无动作返回 0；
        - 汇总并返回长度为 `num_actions` 的计数列表。
        """
        now = time.time()
        cutoff = now - self.window_seconds
        counts = []
        for i in range(self.num_actions):
            ev = self._action_events[i]
            while ev and ev[0] < cutoff:
                ev.popleft()
            if exclude_noop and i == int(self.no_op_action_idx):
                counts.append(0)
            else:
                counts.append(len(ev))
        return counts

    def reset(self, device: torch.device | None = None) -> None:
        """职责：将 DesireThreshold 的内部状态重置到初始配置。

        实现步骤：
        1) 重新分配阈值表并设置初始值；
        2) 清空事件时间戳队列与奖励历史；
        3) 重置生命值评价与最近选择动作；
        4) 可选：迁移到目标设备；
        5) 清空无动作计数并量化；
        """
        # 步骤 1：重建阈值表
        self.thresholds = torch.empty((2, self.num_actions), dtype=torch.float32)
        self.thresholds[0, :] = self.initial_threshold
        self.thresholds_raw = self.thresholds[0, :].clone()
        self.thresholds[1, :] = 0.0
        # 步骤 2：清空事件队列与奖励历史
        for ev in self._action_events:
            ev.clear()
        self.recent_rewards.clear()
        # 步骤 3：重置生命值与最近动作
        self.life_value = 1.0
        if device is not None:
            self.thresholds = self.thresholds.to(device)
        self.last_action = None
        # 步骤 5：清零无动作计数并量化
        self._noop_count = 0
        self._quantize_thresholds()
        dev = device if device is not None else getattr(self, "device", self.thresholds.device)
        self.thresholds = self.thresholds.to(dev)
        self.thresholds_raw = self.thresholds_raw.to(dev)
        self.initial_threshold_vec = self.initial_threshold_vec.to(dev)
        self.cooldown_remaining = self.cooldown_remaining.to(dev)
        self.strong_cooling_remaining = self.strong_cooling_remaining.to(dev)
        self.strong_cooling_step = self.strong_cooling_step.to(dev)

    def save_thresholds(self, file_path: str) -> None:
        """职责：保存 DesireThreshold 的关键阈值状态到 JSON（用于分析/持久化）。

        输出：
        - 写入 `thresholds/thresholds_raw/counts/cooldown_remaining/strong_cooling_remaining/slow_revert_target`。

        实现步骤：
        1) 组装状态字典（含量化阈值与 raw 阈值等）；
        2) 写入 JSON 文件（保留中文）。
        """
        # 步骤 1：组装状态字典
        d = {
            "num_actions": int(self.num_actions),
            "thresholds": [float(v) for v in self.thresholds[0, :].detach().cpu().tolist()],
            "thresholds_raw": [float(v) for v in self.thresholds_raw.detach().cpu().tolist()],
            "counts": [int(self.thresholds[1, i].item()) for i in range(self.num_actions)],
            "cooldown_remaining": [int(self.cooldown_remaining[i].item()) for i in range(self.num_actions)],
            "strong_cooling_remaining": [int(self.strong_cooling_remaining[i].item()) for i in range(self.num_actions)],
            "slow_revert_target": float(self.slow_revert_target),
        }
        # 步骤 2：写入 JSON 文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    def load_thresholds(self, file_path: str) -> None:
        """职责：从 JSON 加载 DesireThreshold 的关键阈值状态（带尺寸校验）。

        实现步骤：
        1) 读取 JSON 文件；
        2) 校验 `num_actions` 尺寸一致；
        3) 恢复 raw/量化阈值并统一量化；
        4) 恢复计数与冷却/强效降温状态、慢回落目标；
        5) 再次量化以确保视图一致。
        """
        # 步骤 1：读取 JSON 文件
        if not os.path.isfile(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # 步骤 2：校验尺寸
        if int(d.get("num_actions", self.num_actions)) != int(self.num_actions):
            raise ValueError("num_actions mismatch")
        # 步骤 3：恢复阈值并量化
        dev = self.thresholds.device
        vals_q = torch.tensor(d.get("thresholds", []), dtype=torch.float32, device=dev)
        vals_r = torch.tensor(d.get("thresholds_raw", []), dtype=torch.float32, device=dev)
        if vals_r.numel() == self.num_actions:
            self.thresholds_raw = torch.clamp(vals_r, min=self.threshold_min, max=self.threshold_max)
        elif vals_q.numel() == self.num_actions:
            self.thresholds_raw = torch.clamp(vals_q, min=self.threshold_min, max=self.threshold_max)
        self._quantize_thresholds()
        # 步骤 4：恢复计数与冷却/强效降温状态
        counts = torch.tensor(d.get("counts", [0]*self.num_actions), dtype=torch.float32, device=dev)
        self.thresholds[1, :] = counts
        cd = torch.tensor(d.get("cooldown_remaining", [0]*self.num_actions), dtype=torch.int32, device=dev)
        self.cooldown_remaining = cd
        scd = torch.tensor(d.get("strong_cooling_remaining", [0]*self.num_actions), dtype=torch.int32, device=dev)
        self.strong_cooling_remaining = scd
        srt = d.get("slow_revert_target", self.slow_revert_target)
        self.slow_revert_target = float(srt)
        # 步骤 5：再量化确保视图一致
        self._quantize_thresholds()

    def update_life_value(self, life_value: float) -> None:
        """职责：更新生命值评价，作为长期追求度计算的输入。

        实现步骤：
        1) 将外部传入的生命值限制在 [0,1] 范围；
        2) 写入到内部状态；
        """
        self.life_value = float(max(0.0, min(1.0, life_value)))

    def _compute_long_term_pursuit(self, life_value: float, avg_reward_recent: float) -> float:
        """职责：计算长期价值追求度阈值，用于负奖励分支的比较。

        实现步骤：
        1) 将生命值限制在 [0,1]；
        2) 将近窗平均奖励限制在 [-1,1] 并归一化到 [0,1]；
        3) 计算乘积权重 p；
        4) 用 `long_term_min` 缩放到负值区间，返回 [long_term_min, 0]；
        """
        l = max(0.0, min(1.0, life_value))
        r = max(-1.0, min(1.0, avg_reward_recent))
        r01 = (r + 1.0) * 0.5
        p = l * r01
        return self.long_term_min * p

    def _quantize_thresholds(self) -> None:
        """职责：将 raw 阈值量化为整数并限制到 [-10,10]，回写到 thresholds[0,:]。"""
        a = self.thresholds_raw
        a = torch.floor(torch.clamp(a, min=self.threshold_min, max=self.threshold_max))
        a = torch.clamp(a, min=-10.0, max=10.0)
        self.thresholds[0, :] = a

# 奖励阈值机制（Reward Threshold）
class RewardThreshold:
    def __init__(
        self,
        num_actions: int, # 动作总数（决定二维结构的第一维）
        num_events: int, # 事件总数（决定二维结构的第二维）
        window_seconds: float = 10.0, # 近窗口统计时长（秒）
        threshold_raise_step: float = 0.15, # 超阈后提升事件阈值的步进值
        slow_decay_interval: float = 5.0, # 事件未触发达到该间隔时触发一次缓降
        slow_decay_step: float = 0.02, # 每次缓降的步进值
        level_count_thresholds: Optional[List[tuple]] = None, # 次数到等级的映射表，例如 [(0,1),(3,2),(6,3)...]
        action_initial_threshold: float = -5.0, # 动作级阈值的初始值
        action_raise_step: float = 0.1, # 动作级阈值随总调整正负变化的步进
        no_op_action_idx: int = 0, # 无操作动作索引，其权重在批量处理中强制置 0
        negative_clamp_threshold: float = -1.0, # 负向钳制的阈值下界，adjusted 落入 [该值,0) 时钳为 0
        global_suppression_duration: float = 5.0, # 全局抑制持续时长（秒）
        global_suppression_amount: float = 0.1, # 全局抑制下对总奖励的固定扣减幅度（绝对值）
        event_threshold_min: float = -2.0, # 事件级阈值缓降下限（可调）
    ) -> None:
        """职责：维护“动作-事件二维”的奖励阈值与动作级阈值，依据近窗频次与奖励强度动态抑制/恢复敏感度，避免奖励数值失真。

        用途：
        - 为每个动作的每个事件维护独立阈值，按近窗触发频次与奖励强度自动升降阈值，抑制过度触发并逐步恢复敏感度；
        - 维护动作级阈值，用于对总奖励进行统一扣减或增益，形成可解释的动作层级控制；
        - 支持等级映射与条件化全局抑制，避免高频事件导致奖励爆发。

        输入参数：
        - num_actions: 动作总数（决定二维结构的第一维）；
        - num_events: 事件总数（决定二维结构的第二维）；
        - window_seconds: 近窗口统计时长（秒）；窗口越大统计越平滑但响应更慢；
        - threshold_raise_step: 超阈后提升事件阈值的步进值；
        - slow_decay_interval: 事件未触发达到该间隔时触发一次缓降；
        - slow_decay_step: 每次缓降的步进值；
        - level_count_thresholds: 次数到等级的映射表，例如 [(0,1),(3,2),(6,3)...]；
        - action_initial_threshold: 动作级阈值的初始值；
        - action_raise_step: 动作级阈值随总调整正负变化的步进；
        - no_op_action_idx: 无操作动作索引，其权重在批量处理中强制置 0；
        - negative_clamp_threshold: 负向钳制的阈值下界，adjusted 落入 [该值,0) 时钳为 0；
        - global_suppression_duration: 全局抑制持续时长（秒）；
        - global_suppression_amount: 全局抑制下对总奖励的固定扣减幅度（绝对值）。

        输出：
        - 无（仅初始化内部状态）。

        实现逻辑步骤：
        1) 记录基础配置与系数；
        2) 初始化当前阈值表、事件队列与阈值历史（二维结构）；
        3) 初始化全局抑制状态与等级结构；
        4) 初始化动作级阈值与上下界；
        5) 设置等级阈值映射。
        """
        self.num_actions = int(num_actions)  # 动作总数（用于二维索引）
        self.num_events = int(num_events)  # 事件总数（用于二维索引）
        self.window_seconds = float(window_seconds)  # 近窗口统计时长（秒）
        self.threshold_raise_step = float(threshold_raise_step)  # 超阈后提升阈值的步进
        self.slow_decay_interval = float(slow_decay_interval)  # 慢速衰减的时间间隔（秒）
        self.slow_decay_step = float(slow_decay_step)  # 每次慢速衰减的步进值
        self.global_suppression_duration = float(global_suppression_duration)  # 全局抑制持续时长（秒）
        self.global_suppression_amount = float(global_suppression_amount)  # 全局抑制下的固定扣减幅度
        self.event_threshold_min = float(event_threshold_min)
        
        # 步骤 2：阈值表、事件队列与历史初始化（每动作持有一套事件，二维结构）
        self.current_thresholds = [0.0 for _ in range(self.num_events)]  # 兼容字段：事件级当前阈值占位
        self.event_queues = [[deque() for _ in range(self.num_events)] for __ in range(self.num_actions)]  # 每动作-事件的时间戳队列
        self.threshold_history = [[[] for _ in range(self.num_events)] for __ in range(self.num_actions)]  # 每动作-事件的历史记录

        # 步骤 3：全局抑制与等级管理
        self.last_any_surpass_ts: Optional[float] = None  # 最近一次任意事件超阈的时间戳
        self.levels_ae = [[0 for _ in range(self.num_events)] for __ in range(self.num_actions)]  # 动作-事件等级矩阵
        self.level_count_thresholds = list(level_count_thresholds) if level_count_thresholds is not None else [  # 次数到等级的映射
            (0, 1),
            (3, 2),
            (6, 3),
            (9, 4),
            (13, 5),
        ]
        self.last_surpass_ts_ae = [[None for _ in range(self.num_events)] for __ in range(self.num_actions)]  # 各动作-事件上次超阈时间
        self.last_occurrence_ts = [None for _ in range(self.num_events)]  # 事件维度的最近出现时间（兼容/预留）
        self.global_suppression_active = False  # 全局抑制状态标志
        self.global_suppression_start_ts = None  # 全局抑制开始时间戳
        self.suppression_decrease_applied = [0.0 for _ in range(self.num_events)]  # 全局抑制下已应用的降幅记录（兼容/预留）

        self.action_thresholds = [float(action_initial_threshold) for _ in range(int(num_actions))]  # 动作级阈值数组
        self.action_raise_step = float(action_raise_step)  # 动作级阈值的调整步进
        self.no_op_action_idx = int(no_op_action_idx)  # 无操作动作索引
        self.action_min = -10.0  # 动作级阈值下界
        self.action_max = 10.0  # 动作级阈值上界
        self.action_event_thresholds = [[0.0 for _ in range(self.num_events)] for __ in range(self.num_actions)]  # 动作-事件阈值矩阵
        self.last_occurrence_ts_ae = [[None for _ in range(self.num_events)] for __ in range(self.num_actions)]  # 各动作-事件最近出现时间戳
        self.negative_clamp_threshold = float(negative_clamp_threshold)  # 负向钳制阈值下界

    def apply(self, action_idx: int, event_idx: int, reward: float, feedback: Optional[float] = None) -> Dict[str, Any]:
        """职责：对一次动作-事件发生进行计数与奖励调整，同时维护等级与全局抑制状态。

        实现步骤：
        1) 更新队列并计算近窗次数；
        2) 按次数阈值映射更新等级；
        3) 达到慢速衰减间隔则降低事件阈值；
        4) 以事件阈值缩减原始奖励并进行小负值钳制；
        5) 超阈上调事件阈值与等级回退；负奖励缓降事件阈值；
        6) 记录历史并返回本次调整结果。
        """
        a = int(action_idx)  # 标准化动作索引
        e = int(event_idx)  # 标准化事件索引
        now = time.time()  # 当前时间戳

        # 步骤 1：事件队列更新与近窗次数计算
        q = self.event_queues[a][e]  # 取出对应动作-事件的时间戳队列
        q.append(now)  # 记录本次发生
        cutoff = now - self.window_seconds  # 近窗口左边界
        while q and q[0] < cutoff:
            q.popleft()  # 清理窗口外的旧记录
        count = len(q)  # 近窗计数

        # 步骤 2：依据次数阈值更新等级
        target_level = 0  # 目标等级初始值
        for c_thr, lvl in self.level_count_thresholds:
            if count >= c_thr:
                target_level = max(target_level, lvl)  # 满足次数阈值则提升目标等级
        self.levels_ae[a][e] = max(self.levels_ae[a][e], target_level)  # 采用就高原则更新等级

        # 步骤 3：慢速衰减——间隔到达时降低事件阈值
        last_ts = self.last_occurrence_ts_ae[a][e]  # 最近一次出现时间戳
        if last_ts is not None and (now - last_ts) >= self.slow_decay_interval:
            self.action_event_thresholds[a][e] = max(self.event_threshold_min, float(self.action_event_thresholds[a][e]) - float(self.slow_decay_step))
        self.last_occurrence_ts_ae[a][e] = now  # 更新最近出现时间戳

        # 步骤 4：用事件阈值缩减原始奖励（得到 adjusted）
        T = float(self.action_event_thresholds[a][e])  # 当前事件阈值
        adjusted = float(reward) - T  # 奖励减去阈值得到调整值

        # 小负值钳制到 0（限定区间 [negative_clamp_threshold, 0)）
        if adjusted < 0.0 and adjusted >= float(self.negative_clamp_threshold):
            adjusted = 0.0  # 小幅负值视为无效

        if float(reward) > T:
            self.action_event_thresholds[a][e] = T + float(self.threshold_raise_step)  # 超阈上调事件阈值
            self.last_any_surpass_ts = now  # 记录任意事件最近超阈时间
            self.last_surpass_ts_ae[a][e] = now  # 记录该动作-事件最近超阈时间
            if self.levels_ae[a][e] > 1:
                self.levels_ae[a][e] -= 1  # 发生超阈时适度回退等级
        elif float(reward) < 0.0:
            self.action_event_thresholds[a][e] = max(self.event_threshold_min, T - float(self.slow_decay_step))

        # 步骤 6：记录历史并返回结果
        self.threshold_history[a][e].append({
            "timestamp": now,  # 记录时间戳
            "count_30s": count,  # 近窗计数
            "threshold": float(self.action_event_thresholds[a][e]),  # 当前事件阈值
            "reward": float(reward),  # 原始奖励
            "adjusted": adjusted,  # 调整后奖励
            "level": self.levels_ae[a][e],  # 当前等级
            "action": a,  # 动作索引
        })

        return {
            "adjusted_reward": adjusted,  # 返回调整后奖励
            "count_30s": count,  # 返回近窗计数
            "threshold": float(self.action_event_thresholds[a][e]),  # 返回当前阈值
            "level": self.levels_ae[a][e],  # 返回当前等级
            "action": a,  # 返回动作索引
        }

    def get_count_30s(self, action_idx: int, event_idx: int) -> int:
        """职责：返回近窗口内某动作-事件的出现次数。

        实现步骤：
        - 清理窗口外的时间戳；
        - 返回队列长度作为近窗计数。
        """
        a = int(action_idx)  # 标准化动作索引
        e = int(event_idx)  # 标准化事件索引
        now = time.time()  # 当前时间戳
        cutoff = now - self.window_seconds  # 近窗口左边界
        q = self.event_queues[a][e]  # 取出时间戳队列
        # 步骤 1：清理窗口外时间戳
        while q and q[0] < cutoff:
            q.popleft()  # 清理过期记录
        # 步骤 2：返回队列长度
        return len(q)  # 返回近窗计数

    def get_history(self, action_idx: int, event_idx: int) -> List[Dict[str, Any]]:
        """职责：获取某动作-事件的阈值与奖励调整历史（按时间顺序）。"""
        return list(self.threshold_history[int(action_idx)][int(event_idx)])  # 返回副本，避免外部修改原始列表

    def reset(self) -> None:
        """职责：重置 RewardThreshold 的内部状态（阈值、队列、历史与全局状态）。

        实现步骤：
        1) 重建当前阈值表；
        2) 清空事件队列与历史；
        3) 清空全局抑制与辅助状态；
        4) 清空最近时间戳；
        """
        # 与初始化保持一致；预留/兼容字段，不参与核心流程
        # 步骤 1：重建当前阈值表
        self.current_thresholds = [0.0 for _ in range(self.num_events)]  # 重建事件级当前阈值占位
        # 步骤 2：清空事件队列与历史
        for a in range(self.num_actions):  # 遍历所有动作
            for e in range(self.num_events):  # 遍历所有事件
                self.event_queues[a][e].clear()  # 清空时间戳队列
                self.threshold_history[a][e].clear()  # 清空历史记录
        # 步骤 3：清空全局抑制与辅助状态
        self.last_any_surpass_ts = None  # 清空任意超阈时间戳
        self.global_suppression_active = False  # 关闭全局抑制状态
        self.global_suppression_start_ts = None  # 清空全局抑制开始时间
        for e in range(self.num_events):
            self.suppression_decrease_applied[e] = 0.0  # 清空已应用的降幅记录
        # 步骤 4：清空最近时间戳
        for e in range(self.num_events):
            self.last_occurrence_ts[e] = None  # 清空事件维度最近出现时间

    def save_thresholds(self, file_path: str) -> None:
        """职责：保存 RewardThreshold 的部分状态到 JSON（用于分析/持久化）。

        内容：
        - `num_events`、`current_thresholds`（预留/兼容）、`levels_ae`（等级结构）。
        """
        # 步骤 1：组装状态字典
        d = {
            "num_events": int(self.num_events),  # 保存事件维度大小
            "current_thresholds": self.current_thresholds,  # 保存当前阈值占位
            "levels_ae": self.levels_ae,  # 保存等级矩阵
        }
        # 步骤 2：写入 JSON 文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)  # 写入 JSON 文件（保留中文）

    def load_thresholds(self, file_path: str) -> None:
        """职责：从 JSON 加载 RewardThreshold 的部分状态（带尺寸校验）。"""
        # 步骤 1：读取 JSON 文件
        if not os.path.isfile(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            d = json.load(f)  # 读取 JSON 文件
        # 步骤 2：尺寸校验
        if int(d.get("num_events", self.num_events)) != int(self.num_events):
            raise ValueError("num_events mismatch")  # 尺寸不一致直接报错
        ct = d.get("current_thresholds", self.current_thresholds)  # 读取当前阈值占位
        lv_ae = d.get("levels_ae", None)  # 读取等级矩阵
        if not isinstance(ct, list):
            raise ValueError("invalid thresholds file format")  # 格式校验
        # 步骤 3：恢复当前阈值占位
        self.current_thresholds = [float(x) for x in ct]  # 恢复当前阈值占位（浮点）
        # 步骤 4：恢复等级矩阵
        if lv_ae is not None and isinstance(lv_ae, list):
            self.levels_ae = [[int(x) for x in row] for row in lv_ae]  # 恢复等级矩阵（整型）
        else:
            self.levels_ae = [[0 for _ in range(self.num_events)] for __ in range(self.num_actions)]  # 尺寸一致但缺失时回退到 0

    def apply_many(self, action_idx: int, events: List[Dict[str, Any]], action_weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """职责：按动作权重批量处理事件，更新各动作-事件阈值并计算总调整。

        输入：
        - `action_idx`: 当前主动作索引（用于默认权重赋值）。
        - `events`: 若干 `{event, reward}` 项；将对所有非零权重动作应用事件影响。
        - `action_weights`: 动作权重分布；为空时默认主动作为 1，其余为 0；无操作动作权重强制置 0。

        实现步骤：
        1) 逐事件遍历各动作，记录事件发生到队列并计算加权阈值和；
        2) 超阈则提升事件阈值并下调等级；负奖励则缓降事件阈值；
        3) 对总调整应用动作级扣减并更新动作级阈值；
        4) 依据近窗计数映射等级，条件化进入全局抑制；
        5) 对事件阈值进行缓降（按等级步长）；
        6) 返回总调整与细节。

        输出：
        - `{adjusted_total, details}`：总调整与逐事件细节（包含加权阈值和与调整值）。
        """
        total = 0.0  # 累计总调整值
        details: List[Dict[str, Any]] = []  # 逐事件细节
        now = time.time()  # 当前时间戳
        # 步骤 1：准备动作权重
        weights = list(action_weights) if action_weights is not None else None  # 标准化权重列表
        if weights is None:
            weights = [0.0 for _ in range(self.num_actions)]  # 默认将所有权重置 0
            weights[int(action_idx)] = 1.0  # 主动作置为 1
        if int(self.no_op_action_idx) < len(weights):
            weights[int(self.no_op_action_idx)] = 0.0  # 无操作动作权重强制为 0
        # 步骤 2：逐事件计算加权阈值和，并按 rw/Tie 更新事件阈值与调整值
        for item in (events or []):  # 遍历输入事件
            e = int(item.get("event", 0))  # 事件索引
            rw = float(item.get("reward", 0.0))  # 事件奖励
            sumT = 0.0  # 加权阈值和
            for i in range(self.num_actions):  # 遍历动作
                wi = float(weights[i])  # 当前动作权重
                if wi <= 0.0:
                    continue  # 零或负权重跳过
                Tie = float(self.action_event_thresholds[i][e])  # 当前动作-事件阈值
                sumT += wi * Tie  # 累计加权阈值
                self.last_occurrence_ts_ae[i][e] = now  # 更新最近出现时间戳
                if rw > Tie:
                    self.action_event_thresholds[i][e] = float(Tie) + wi * float(self.threshold_raise_step)  # 超阈上调，受权重影响
                    self.last_surpass_ts_ae[i][e] = now  # 记录最近超阈时间戳
                    if self.levels_ae[i][e] > 1:
                        self.levels_ae[i][e] -= 1  # 超阈时等级回退
                elif rw < 0.0:
                    self.action_event_thresholds[i][e] = max(self.event_threshold_min, float(Tie) - wi * float(self.slow_decay_step))
            adj = rw - sumT  # 奖励减去加权阈值和得到调整值
            if adj < 0.0 and adj >= float(self.negative_clamp_threshold):
                adj = 0.0  # 小负值钳制
            total += adj  # 累加总调整
            details.append({"event": e, "reward": rw, "sum_threshold": sumT, "adjusted_reward": adj})  # 记录细节

        # 步骤 3：动作级惩罚与动作阈值更新
        penalty = 0.0  # 动作级扣减累计
        for i in range(self.num_actions):
            wi = float(weights[i])  # 当前动作权重
            if wi <= 0.0:
                continue  # 跳过零权重
            if i == int(self.no_op_action_idx):
                continue  # 无操作动作不参与扣减
            aT = float(self.action_thresholds[i])  # 动作级阈值
            penalty += wi * aT  # 累计加权扣减
        total = float(total) - float(penalty)  # 应用动作级扣减
        for i in range(self.num_actions):  # 更新动作级阈值
            wi = float(weights[i])  # 当前动作权重
            if wi <= 0.0:
                continue  # 跳过零权重
            if i == int(self.no_op_action_idx):
                continue  # 无操作动作不参与更新
            aT = float(self.action_thresholds[i])  # 当前动作级阈值
            if total >= 0.0:
                self.action_thresholds[i] = min(self.action_max, aT + wi * self.action_raise_step)  # 正向总调整时上调动作阈值
            else:
                self.action_thresholds[i] = max(self.action_min, aT - wi * self.action_raise_step)  # 负向总调整时下调动作阈值

        # 步骤 4：事件等级与条件化全局抑制 + 事件阈值缓降
        for e in range(self.num_events):  # 遍历事件维度
            cond_any = False  # 是否触发全局抑制的条件
            step_per_e = []  # 各动作对该事件的缓降步长
            for i in range(self.num_actions):
                # 依据近窗计数映射等级
                q = self.event_queues[i][e]  # 取出队列
                cutoff = now - self.window_seconds  # 近窗口左边界
                while q and q[0] < cutoff:
                    q.popleft()  # 清理过期时间戳
                cnt = len(q)  # 近窗计数
                lvl_i = 0  # 等级初值
                for c_thr, lvl in self.level_count_thresholds:
                    if cnt >= c_thr:
                        lvl_i = max(lvl_i, lvl)  # 满足次数阈值则提升等级
                step_per_e.append(0.05 if lvl_i <= 1 else 0.1)
                last_surpass_i = self.last_surpass_ts_ae[i][e]  # 最近超阈时间戳
                if lvl_i > 2 and (last_surpass_i is None or (now - last_surpass_i) >= self.global_suppression_duration):
                    cond_any = True  # 等级高且超阈不频繁则触发全局抑制条件
            if cond_any:
                total = float(total) - float(self.global_suppression_amount)  # 应用全局抑制扣减
                self.global_suppression_active = True  # 标记进入全局抑制
                if self.global_suppression_start_ts is None:
                    self.global_suppression_start_ts = now  # 记录抑制开始时间
            for i in range(self.num_actions):  # 事件阈值缓降（按等级步长）
                last_ts = self.last_occurrence_ts_ae[i][e]  # 最近出现时间戳
                if last_ts is None:
                    continue  # 尚未出现不缓降
                if (now - last_ts) >= 1.0:
                    v = float(self.action_event_thresholds[i][e])  # 当前阈值
                    step = step_per_e[i] if i < len(step_per_e) else 0.5  # 缓降步长选取
                    self.action_event_thresholds[i][e] = max(self.event_threshold_min, v - step)

        # 步骤 6：返回总调整与细节
        return {"adjusted_total": float(total), "details": details}  # 返回总调整与细节

 



if __name__ == "__main__":
    rt = RewardThreshold(
        num_actions=9,
        num_events=9,
        window_seconds=10.0,
        threshold_raise_step=0.1,
        slow_decay_interval=3.0,
        slow_decay_step=0.1,
        action_initial_threshold=-5.0,
        action_raise_step=0.1,
        no_op_action_idx=0,
        negative_clamp_threshold=-2.0,
    )
    events = [
        {"event": 5, "reward": 3.0},
        {"event": 2, "reward": -1.0},
        {"event": 3, "reward": 0.5},
    ]
    weights = [0.0 for _ in range(9)]
    weights[1] = 0.6
    weights[2] = 0.4
    res = rt.apply_many(action_idx=1, events=events, action_weights=weights)
    print("details:", res["details"]) 
    print("adjusted_total:", res["adjusted_total"])
    ae = rt.action_event_thresholds
    print("T[action,event]:", ae[1][5], ae[2][5])