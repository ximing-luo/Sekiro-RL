# DesireThreshold 阈值表与升降机制

- 阈值表
  - `thresholds[0,:]`：量化兴奋阈值（整数，范围 [-10,10]）
  - `thresholds[1,:]`：近窗口选择次数（按事件时间戳统计）
  - `thresholds_raw`：连续兴奋阈值（真实来源，用于所有数值更新）
  - `initial_threshold_vec`：每动作初始阈值向量（基线）
  - 附属状态：`cooldown_remaining`、`strong_cooling_remaining`、`strong_cooling_step`

- 升降特效机制
  - 强烈抬高（Spike）：非无动作被选后抬高至 `initial + spike_amount`，并启动 `cooldown_remaining` 按 `spike_decay_step` 回落
  - 冷却回落：处于冷却的动作逐步回落至初始值，计数器逐步递减
  - 强效降温：由无动作/饥饿触发，对 Top-K 或长期未选的动作按步均幅下降（`strong_cooling_duration_steps`/`strong_cooling_step`）
  - 自动缓慢回落：周期性将所有动作阈值向 `slow_revert_target` 轻微靠近（排除无动作索引），并保持量化同步
  - 奖励分支：
    - 正奖励：低频降阈，高频按溢出量 `overflow * inhibition_coeff` 升阈
    - 负奖励：与长期追求阈值比较，劣于目标升阈抑制重复错误，否则降阈鼓励探索
  - 无动作专属：主动选择分级升阈；其他动作被选后小幅回落；被动选择过多触发全局强效降温；每步向初始值归位

# RewardThreshold 阈值表与升降机制

- 阈值表
  - `action_event_thresholds[a][e]`：动作-事件阈值矩阵（事件级）
  - `event_queues[a][e]`：时间戳队列，用于近窗计数
  - `levels_ae[a][e]`：动作-事件等级矩阵
  - `action_thresholds[a]`：动作级阈值数组（用于总惩罚与调节）
  - 其他：`threshold_history[a][e]`、`last_surpass_ts_ae`、`last_occurrence_ts_ae`、`global_suppression_active/start_ts`

- 升降特效机制
  - 事件超阈上调：`reward > T` 时按 `threshold_raise_step` 上调阈值，并回退等级
  - 事件负奖励缓降：`reward < 0` 时按 `slow_decay_step` 缓降阈值（不低于 -5）
  - 事件慢速衰减：未触发超过 `slow_decay_interval` 或超过 1s 按等级步长（`0.5/1.0`）缓降
  - 动作级惩罚：按权重扣减总调整后，对 `action_thresholds` 随总调整正负进行升/降（`action_raise_step`，范围 [-10,10]）
  - 全局抑制：高等级且超阈不频繁时统一扣减 `global_suppression_amount` 并进入抑制期
  - 小负值钳制：调整值落入 `[negative_clamp_threshold, 0)` 时钳为 0（影响奖励缩放，不更改阈值）