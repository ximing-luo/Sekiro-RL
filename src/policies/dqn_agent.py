"""
模块用途：DQN 代理实现，负责动作选择与网络训练优化。

包含：
- 函数：_np_to_torch_imgs(np_batch)
- 类：DQNAgent（select_action/optimize/update_target/save_model/load_model）

边界：
- 负责：前向推理与参数更新（包含目标网络同步与模型存取）
- 不负责：环境交互、采集与可视化、事件分类与指标写入（均在外部模块）
"""

import time
import threading
import os
import sys
import torch

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch.nn as nn
import torch.nn.functional as F
import contextlib
from src.models.simple_dqn import dqn_simple
from src.models.resnet import ddqn_res18
from src.envs.tasks.sekiro.action_map import no_op_index
import configs.config as config

# 设备选择：优先使用 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

FRAME_HISTORY_LEN = config.FRAME_HISTORY_LEN  # 历史帧堆叠长度，决定输入通道数
GAMMA = config.GAMMA  # 折扣因子，用于计算目标Q值（从配置读取）
LR = config.LR  # 学习率，从配置读取
TARGET_UPDATE_FREQ = config.TARGET_UPDATE_FREQ  # 目标网络同步频率（按训练步计数）
MICRO_BATCH_SIZE = config.MICRO_BATCH_SIZE  # 单次采样的最小批大小（用于采样与AMP，配置项）
BATCH_SIZE = config.BATCH_SIZE  # 逻辑训练批大小（影响梯度累积，配置项）
GRAD_CLIP_NORM = getattr(config, 'GRAD_CLIP_NORM', 10.0)
PER_ALPHA = getattr(config, 'PER_ALPHA', 0.6)
PER_BETA_START = getattr(config, 'PER_BETA_START', 0.4)
PER_BETA_END = getattr(config, 'PER_BETA_END', 1.0)
PER_BETA_STEPS = getattr(config, 'PER_BETA_STEPS', 200000)
GRAD_ACCUM_STEPS = max(1, BATCH_SIZE // MICRO_BATCH_SIZE)  # 梯度累积步数


def _np_to_torch_imgs(np_batch):
    """将 numpy 图像批次 (N, C, H, W) 转换为 torch.float32 并归一化到 [0,1]。"""
    t = torch.from_numpy(np_batch)
    if device.type == "cuda":
        try:
            t = t.pin_memory()
            return t.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
        except Exception:
            return t.to(device=device, dtype=torch.float32, non_blocking=False) / 255.0
    else:
        return t.to(device=device, dtype=torch.float32) / 255.0


class DQNAgent:
    """Dueling DQN 代理：负责动作选择、优化训练、模型保存与加载。"""

    def __init__(self, img_width, img_height, action_dim, model_file=None, n_step_rewards: int = 1):
        in_channels = FRAME_HISTORY_LEN
        self.action_dim = action_dim
        self.model_file = model_file or config.MODEL_PATH
        self.n_step_rewards = int(max(1, n_step_rewards))

        # 网络模块：在线/目标/训练
        self.eval_net = ddqn_res18(in_channels=in_channels, num_actions=action_dim).to(device)
        self.target_net = ddqn_res18(in_channels=in_channels, num_actions=action_dim).to(device)
        self.train_net = ddqn_res18(in_channels=in_channels, num_actions=action_dim).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.train_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        try:
            self.eval_net.eval()
        except Exception:
            pass

        # 训练加速：CUDA 流
        if device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                low_pri, high_pri = props.stream_priorities_range
                self.train_stream = torch.cuda.Stream(priority=low_pri)
            except Exception:
                self.train_stream = torch.cuda.Stream()

        # 加载预训练模型（如果存在）并冻结主干
        # 注意：这里假设预训练模型是单通道输入，而当前模型可能是多通道（Frame Stack）
        # 我们会将单通道权重复制扩展到多通道
        pretrained_path = os.path.join(project_root, "models", "resnet_model.pth")
        if os.path.isfile(pretrained_path):
            self._load_pretrained_backbone(pretrained_path, in_channels)

        # 优化器与损失函数（Huber Loss 更稳定）
        # 仅优化 requires_grad=True 的参数（即未冻结的层）
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.train_net.parameters()), lr=LR)
        self.criterion = nn.SmoothL1Loss()

        # 训练状态与并发控制
        self.train_steps = 0  # 训练步计数（用于目标网络同步等周期控制）
        self.optimize_count = 0  # 已完成的优化次数（线程级优化步统计）
        self.last_loss = None  # 最近一次损失值缓存（便于日志与监控）
        self.optimizer_lock = threading.Lock()  # 优化过程互斥锁，防止并发调用 optimize
        self.grad_accum_steps = GRAD_ACCUM_STEPS  # 梯度累积需要的步数阈值
        self.grad_accum_count = 0  # 当前已累积的步数计数（达到阈值触发更新）
        if device.type == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')  # AMP 梯度缩放器，提高混合精度稳定性
        self.update_lock = threading.Lock()  # 将 train_net 参数复制到 eval_net 的同步锁
        self._step_in_progress = False  # 标记 optimizer.step 是否正在进行（用于并发安全）

        # 移除了动机/奖励模块（excitability_threshold.py）

        # 运行元数据与缓存
        self.run_id = time.strftime('%Y%m%d-%H%M%S')  # 本次训练运行唯一标识（用于日志命名）
        self._last_q = None  # 最近一次前向计算的原始 Q 值缓存

    def _load_pretrained_backbone(self, model_path, current_in_channels):
        """加载预训练 ResNet 权重，适配多通道输入，并冻结主干层。"""
        print(f"正在加载预训练主干网络: {model_path}")
        try:
            pretrained_dict = torch.load(model_path, map_location=device)
            # 如果是保存的整个 state_dict，直接使用；如果是模型对象，取 state_dict
            if hasattr(pretrained_dict, 'state_dict'):
                pretrained_dict = pretrained_dict.state_dict()
            
            model_dict = self.eval_net.state_dict()
            
            # 过滤并适配参数
            new_state_dict = {}
            for k, v in pretrained_dict.items():
                # 跳过全连接层（因为 num_actions 可能不同）
                if "fc_adv" in k or "fc_val" in k:
                    continue
                
                # 适配第一层卷积（conv1[0]）：深度卷积 (in=1 -> in=N)
                # 原始形状: (1, 1, 3, 3) -> 目标形状: (N, 1, 3, 3)
                # 注意：Dueling_DQN 定义 conv1[0] groups=in_channels
                if k == "conv1.0.weight":
                    if v.shape[0] == 1 and current_in_channels > 1:
                        print(f"适配 {k}: {v.shape} -> ({current_in_channels}, 1, 3, 3)")
                        v = v.repeat(current_in_channels, 1, 1, 1)
                
                # 适配第二层卷积（conv1[1]）：点卷积 (in=1 -> in=N)
                # 原始形状: (64, 1, 1, 1) -> 目标形状: (64, N, 1, 1)
                elif k == "conv1.1.weight":
                    if v.shape[1] == 1 and current_in_channels > 1:
                        print(f"适配 {k}: {v.shape} -> (64, {current_in_channels}, 1, 1)")
                        v = v.repeat(1, current_in_channels, 1, 1)
                        # 这里我们选择直接复制，相当于把所有帧的特征求和
                
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        new_state_dict[k] = v
                    else:
                        print(f"形状不匹配跳过 {k}: 预训练 {v.shape} vs 模型 {model_dict[k].shape}")
            
            # 更新权重
            model_dict.update(new_state_dict)
            self.eval_net.load_state_dict(model_dict)
            print("预训练主干网络加载完成。")
            
            # 冻结主干网络参数
            print("正在冻结主干网络参数...")
            frozen_count = 0
            trainable_count = 0
            for name, param in self.eval_net.named_parameters():
                if "fc_adv" in name or "fc_val" in name:
                    param.requires_grad = True
                    trainable_count += 1
                    # print(f"  保持可训练: {name}")
                else:
                    param.requires_grad = False
                    frozen_count += 1
            print(f"已冻结 {frozen_count} 个参数张量，保留 {trainable_count} 个可训练参数张量（fc_adv/fc_val）。")
            
            # 同步到 target_net 和 train_net，并设置 train_net 的 requires_grad
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.train_net.load_state_dict(self.eval_net.state_dict())
            
            for name, param in self.train_net.named_parameters():
                if "fc_adv" in name or "fc_val" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            import traceback
            traceback.print_exc()

    def select_action(self, state_tensor, epsilon=0.0):
        # Epsilon-Greedy 策略
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()

        with torch.no_grad():
            self.update_lock.acquire()
            try:
                q_values = self.eval_net(state_tensor)
                # 直接选择 Q 值最大的动作
                action = torch.argmax(q_values).item()
                try:
                    self._last_q = q_values[0].detach().cpu().numpy()
                except Exception:
                    pass
            finally:
                self.update_lock.release()
        return action


    def optimize(self, replay_buffer):
        """从环境的经验缓冲区采样并进行一次优化（在单独线程中运行）。
        流程分层说明（仅注释，逻辑不改）：
        1) 前置检查与并发控制：确认可采样，获取优化锁
        2) 采样与张量化：一次性取出小批量并转换到设备
        3) 前向与目标计算：用在线训练网计算 `Q(s,a)`，用目标网计算 Bellman 目标
        4) 损失与反向：用 `self.criterion` 计算损失，支持梯度累积与 AMP 反向
        5) 参数更新与同步：触发 `optimizer.step`，同步 `train_net -> eval_net`
        6) 日志与周期性目标网络同步：按 `TARGET_UPDATE_FREQ` 将 `eval_net -> target_net`
        """
        if not replay_buffer.can_sample(MICRO_BATCH_SIZE):
            return
        if self.n_step_rewards > 1:
            if not hasattr(replay_buffer, 'can_sample_n_step') or not replay_buffer.can_sample_n_step(MICRO_BATCH_SIZE, self.n_step_rewards):
                return

        # === 阶段 1：前置检查与并发控制 ===
        # 使用锁确保同一时间只有一个线程在优化
        if not self.optimizer_lock.acquire(blocking=False):
            return  # 如果锁已被占用，则直接返回，避免阻塞

        try:
            # === 阶段 2：采样与基本说明 ===
            # 采样 (obs, act, rew, next_obs, done_mask)
            # - obs                形状: (N, C, H, W)
            # - act                形状: (N,)
            # - rew                形状: (N,)
            # - next_obs           形状: (N, C, H, W)
            # - done_mask          形状: (N,), 值为 {0.0, 1.0}，1.0 表示回合结束
            use_per = False
            beta_now = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * min(1.0, float(self.optimize_count) / float(max(1, PER_BETA_STEPS)))
            if self.n_step_rewards > 1 and hasattr(replay_buffer, 'sample_n_step_per'):
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, steps_used, idxes, weights = replay_buffer.sample_n_step_per(MICRO_BATCH_SIZE, self.n_step_rewards, GAMMA, alpha=PER_ALPHA, beta=beta_now)
                use_per = True
            elif self.n_step_rewards == 1 and hasattr(replay_buffer, 'sample_per'):
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, idxes, weights = replay_buffer.sample_per(MICRO_BATCH_SIZE, alpha=PER_ALPHA, beta=beta_now)
                use_per = True
            else:
                if self.n_step_rewards > 1:
                    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, steps_used = replay_buffer.sample_n_step(MICRO_BATCH_SIZE, self.n_step_rewards, GAMMA)
                else:
                    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(MICRO_BATCH_SIZE)

            # === 阶段 3：张量化与设备迁移 ===
            # 转换为 torch 张量并归一化到 [0,1]（像素值除以 255）
            # 注意：此处是对整个批次做一次性转换，非逐样本循环
            with (torch.cuda.stream(self.train_stream) if device.type == "cuda" else contextlib.nullcontext()):
                obs_batch_t = _np_to_torch_imgs(obs_batch)
                next_obs_batch_t = _np_to_torch_imgs(next_obs_batch)
                act_batch_t = torch.from_numpy(act_batch).to(device=device, dtype=torch.long)
                rew_batch_t = torch.from_numpy(rew_batch).to(device=device, dtype=torch.float32)
                done_mask_t = torch.from_numpy(done_mask).to(device=device, dtype=torch.float32)
                if use_per:
                    w_t = torch.from_numpy(weights).to(device=device, dtype=torch.float32)

                # === 阶段 4：前向与 Q(s,a) 选取 ===
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        q_values = self.train_net(obs_batch_t)
                        q_sa = q_values.gather(1, act_batch_t.view(-1, 1)).squeeze(1)
                else:
                    q_values = self.train_net(obs_batch_t)
                    q_sa = q_values.gather(1, act_batch_t.view(-1, 1)).squeeze(1)

                # === 阶段 5：Bellman 目标计算（no_grad） ===
                # 用目标网络得到 next_state 的最大 Q 值，构造 TD 目标：
                # target = r + gamma * (1 - done) * max_a' Q_target(s', a')
                # no_grad：禁用自动求导，避免构图与梯度记录，降低开销且不更新 target_net
                with torch.no_grad():
                    next_q_eval = self.eval_net(next_obs_batch_t)
                    next_act = next_q_eval.argmax(1)
                    next_q_tgt = self.target_net(next_obs_batch_t)
                    next_q_sa = next_q_tgt.gather(1, next_act.view(-1, 1)).squeeze(1)
                    if self.n_step_rewards > 1:
                        steps_used_t = torch.from_numpy(steps_used).to(device=device, dtype=torch.float32)
                        gamma_power = torch.pow(torch.tensor(GAMMA, dtype=torch.float32, device=device), steps_used_t)
                        target = rew_batch_t + gamma_power * (1.0 - done_mask_t) * next_q_sa
                    else:
                        target = rew_batch_t + GAMMA * (1.0 - done_mask_t) * next_q_sa

                # === 阶段 6：损失与反向传播 ===
                # 损失函数位置：此行即为损失计算。
                # 预测值：q_sa（在线训练网对实际动作 a 的 Q 值）
                # 目标值：target（由目标网与 Bellman 方程计算）
                # 损失函数来源：在 __init__ 中设置为 `self.criterion = nn.SmoothL1Loss()`，
                # 如需改为 MSE，可在 __init__ 中替换为 nn.MSELoss。
                if use_per:
                    loss_vec = F.smooth_l1_loss(q_sa, target, reduction='none')
                    loss = (w_t * loss_vec).mean()
                else:
                    loss = self.criterion(q_sa, target)
                loss = loss / float(self.grad_accum_steps)
                if self.grad_accum_count == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    self.scaler.scale(loss).backward()
                    try:
                        self.scaler.unscale_(self.optimizer)
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(self.train_net.parameters(), max_norm=float(GRAD_CLIP_NORM))
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.train_net.parameters(), max_norm=float(GRAD_CLIP_NORM))
                self.grad_accum_count += 1

                # === 阶段 7：触发参数更新与 eval_net 同步 ===
                stepped = False
                if self.grad_accum_count >= self.grad_accum_steps:
                    self._step_in_progress = True
                    if device.type == "cuda":
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.update_lock.acquire()
                    try:
                        self.eval_net.load_state_dict(self.train_net.state_dict())
                    finally:
                        self.update_lock.release()
                        self._step_in_progress = False
                    self.grad_accum_count = 0
                    stepped = True
                    self.train_steps += 1
                    self.optimize_count += 1
                    if hasattr(self, 'update_target_soft'):
                        self.update_target_soft(tau=0.005)
            self.last_loss = float(loss.item())
            if stepped:
                print(f"\033[91m***已完成优化线程，当前优化步数：{self.optimize_count}，当前损失：{self.last_loss:.4f}***\033[0m")
            if use_per and hasattr(replay_buffer, 'update_priorities'):
                try:
                    td_err = (q_sa.detach() - target.detach()).abs().cpu().numpy()
                    replay_buffer.update_priorities(idxes, td_err)
                except Exception:
                    pass

            # === 阶段 8：周期性目标网络同步 ===
            # 周期性同步目标网络参数：将 eval_net 的最新参数复制到 target_net
            # 使用训练步数作为周期计数，频率由 TARGET_UPDATE_FREQ 控制
            if self.train_steps % TARGET_UPDATE_FREQ == 0:
                self.update_target()
        finally:
            # === 阶段 9：释放优化锁 ===
            # 释放优化锁，允许后续优化线程进入
            self.optimizer_lock.release()

    def update_target(self):
        """将目标网络参数同步为在线网络参数。"""
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def update_target_soft(self, tau=0.005):
        with torch.no_grad():
            for p_tgt, p_eval in zip(self.target_net.parameters(), self.eval_net.parameters()):
                p_tgt.data.mul_(1.0 - float(tau)).add_(float(tau) * p_eval.data)

    def save_model(self):
        """保存在线网络参数到文件。"""
        # 确保目录存在
        dirpath = os.path.dirname(self.model_file)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        torch.save(self.eval_net.state_dict(), self.model_file)

    def load_model(self):
        """从文件加载在线网络参数，并同步到目标网络。"""
        state_dict = torch.load(self.model_file, map_location=device)
        self.eval_net.load_state_dict(state_dict)
        try:
            self.train_net.load_state_dict(self.eval_net.state_dict())
        except Exception:
            pass
        self.update_target()
        print("模型加载成功。")



if __name__ == "__main__":
    pass



