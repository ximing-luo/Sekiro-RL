"""
模块用途：训练过程的度量写入与可选的实时可视化工具。

包含：
- 函数：write_csv(...), write_json(...)
- 类：VizRunner（start/stop）、InputVisRunner（start/update/stop）

边界：
- 负责：数据持久化与图形展示
- 不负责：训练逻辑、环境交互与模型管理
"""
import os
import sys

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import csv
import json
import time as _time
import time
import threading
import numpy as np
import cv2
from src.interfaces.system import window as window_utils
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

def write_csv(log_dir, run_id, step, episode, action, events_feedback, adj_reward, last_q, last_q_mod, desire_thresholds_row=None, reward_thresholds_action=None, raw_reward=None, adv_values=None, adv_values_shrink=None, state_value=None, events=None, noop_1s_count=None, action_recent_counts=None, reward_avg_recent=None, epsilon=None):
    """职责：将训练关键度量以 CSV 形式追加到日志文件。

    实现步骤：
    1) 确保日志目录存在；
    2) 构造写入行的字段；
    3) 判断是否需要写入表头；
    4) 以追加模式写入一行；

    函数用来干什么：将一条训练过程的统计数据（动作、事件、奖励与阈值快照）写入 CSV，便于后续分析。
    参数说明（输入/输出）：
    - 输入：
      - `log_dir`（str）：日志目录
      - `run_id`（str）：本次运行标识
      - `step`（int）：训练步编号
      - `action`（int）：选择的动作索引
      - `event_idx`（int）：事件索引
      - `fb`（float）：事件反馈值
      - `adj_reward`（float）：调整后的奖励
      - `last_q`（array-like|None）：最近一次原始 Q 值
      - `last_q_mod`（array-like|None）：最近一次加权后的 Q 值
      - `desire_thresholds_row`（list[float]）：当前欲望阈值的第一行快照
      - `reward_thresholds_action`（list[float]）：当前动作的事件阈值数组
    - 输出：无（写文件副作用）
    """
    # 步骤 1：确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'train_metrics.csv')
    # 步骤 2：构造数据行字段
    evfb_str = '|'.join([f"{int(e)}:{float(fb):.6f}" for e, fb in (events_feedback or [])])
    events_str = ';'.join([f"{int(e)}" for e in (events or [])])
    row = {
        'run_id': run_id,
        'step': int(step),
        'episode': int(episode),
        'action': int(action),
        'events_feedback': evfb_str,
        'adj_reward': float(adj_reward),
        'raw_reward': (float(raw_reward) if raw_reward is not None else None),
        'reward_avg_recent': (float(reward_avg_recent) if reward_avg_recent is not None else None),
        'q_values': ';'.join([f"{float(x):.6f}" for x in (list(last_q) if last_q is not None else [])]),
        'q_values_mod': ';'.join([f"{float(x):.6f}" for x in (list(last_q_mod) if last_q_mod is not None else [])]),
        'desire_thresholds': ';'.join([f"{float(x):.6f}" for x in (list(desire_thresholds_row) if desire_thresholds_row is not None else [])]),
        'reward_thresholds_action': ';'.join([f"{float(x):.6f}" for x in (list(reward_thresholds_action) if reward_thresholds_action is not None else [])]),
        'adv_values': ';'.join([f"{float(x):.6f}" for x in (list(adv_values) if adv_values is not None else [])]),
        'adv_values_shrink': ';'.join([f"{float(x):.6f}" for x in (list(adv_values_shrink) if adv_values_shrink is not None else [])]),
        'state_value': (float(state_value) if state_value is not None else None),
        'events': events_str,
        'noop_1s_count': (int(noop_1s_count) if noop_1s_count is not None else None),
        'action_recent_counts': ';'.join([str(int(x)) for x in (list(action_recent_counts) if action_recent_counts is not None else [])]),
        'epsilon': (float(epsilon) if epsilon is not None else None),
    }
    # 步骤 3：是否需要写入表头
    need_header = not os.path.exists(path)
    # 步骤 4：以追加模式写入行
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if need_header:
            w.writeheader()
        w.writerow(row)

def write_json(log_dir, run_id, step, episode, action, events_feedback, adj_reward, last_q, last_q_mod, desire_thresholds_row=None, reward_thresholds_action=None, reward_thresholds_full=None, fps=None, adv_values=None, state_value=None, events=None, raw_reward=None, noop_1s_count=None, adv_values_shrink=None, feat_input_b64=None, feat_static_b64=None, feat_dynamic_b64=None, f0_last_maps_b64=None, static_layers_b64=None, dynamic_layers_b64=None, action_recent_counts=None, reward_avg_recent=None, epsilon=None):
    """职责：将最近一次训练度量写入 JSON 文件，便于前端读取与可视化。

    实现步骤：
    1) 确保日志目录存在；
    2) 整理并转换各字段类型；
    3) 组装 payload 数据结构；
    4) 写入 `latest.json`；

    函数用来干什么：以结构化 JSON 的形式输出最新一次训练状态，供 Streamlit/前端或其他工具读取。
    参数说明（输入/输出）：
    - 输入：参数同 `write_csv`，另包含：
      - `reward_thresholds_full`（list[list[float]]）：所有动作的事件阈值矩阵快照
    - 输出：无（写文件副作用）
    """
    # 步骤 1：确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    # 步骤 2：整理并转换各字段类型
    q_vals = [float(x) for x in (list(last_q) if last_q is not None else [])]
    q_mods = [float(x) for x in (list(last_q_mod) if last_q_mod is not None else [])]
    des_row = [float(x) for x in (list(desire_thresholds_row) if desire_thresholds_row is not None else [])]
    r_act = [float(x) for x in (list(reward_thresholds_action) if reward_thresholds_action is not None else [])]
    r_full = [[float(x) for x in row] for row in (list(reward_thresholds_full) if reward_thresholds_full is not None else [])]
    adv_vals = [float(x) for x in (list(adv_values) if adv_values is not None else [])]
    adv_vals_shrink = [float(x) for x in (list(adv_values_shrink) if adv_values_shrink is not None else [])]
    # 步骤 3：组装 payload
    payload = {
        'run_id': run_id,
        'step': int(step),
        'episode': int(episode),
        'action': int(action),
        'events_feedback': [[int(e), float(fb)] for e, fb in (events_feedback or [])],
        'events': [int(e) for e in (list(events) if events is not None else [])],
        'adj_reward': float(adj_reward),
        'raw_reward': (float(raw_reward) if raw_reward is not None else None),
        'reward_avg_recent': (float(reward_avg_recent) if reward_avg_recent is not None else None),
        'q_values': q_vals,
        'q_values_mod': q_mods,
        'adv_values': adv_vals,
        'adv_values_shrink': adv_vals_shrink,
        'state_value': (float(state_value) if state_value is not None else None),
        'desire_thresholds': des_row,
        'reward_thresholds_action': r_act,
        'reward_thresholds_full': r_full,
        'fps': (float(fps) if fps is not None else None),
        'noop_1s_count': (int(noop_1s_count) if noop_1s_count is not None else None),
        'action_recent_counts': [int(x) for x in (list(action_recent_counts) if action_recent_counts is not None else [])],
        'feat_input_b64': feat_input_b64,
        'feat_static_b64': feat_static_b64,
        'feat_dynamic_b64': feat_dynamic_b64,
        'f0_last_maps_b64': (list(f0_last_maps_b64) if f0_last_maps_b64 is not None else None),
        'static_layers_b64': (list(static_layers_b64) if static_layers_b64 is not None else None),
        'dynamic_layers_b64': (list(dynamic_layers_b64) if dynamic_layers_b64 is not None else None),
        'epsilon': (float(epsilon) if epsilon is not None else None),
    }
    # 步骤 4：写入文件
    path = os.path.join(log_dir, 'latest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

def write_images_json(log_dir, step, f0_last_maps_b64=None, static_layers_b64=None, dynamic_layers_b64=None):
    os.makedirs(log_dir, exist_ok=True)
    payload = {
        'step': int(step),
        'f0_last_maps_b64': (list(f0_last_maps_b64) if f0_last_maps_b64 is not None else None),
        'static_layers_b64': (list(static_layers_b64) if static_layers_b64 is not None else None),
        'dynamic_layers_b64': (list(dynamic_layers_b64) if dynamic_layers_b64 is not None else None),
    }
    path = os.path.join(log_dir, 'latest_images.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

class VizRunner:
    """职责：实时绘制一次推理的 Q 值与阈值条形图，以及事件阈值与事件反馈强度。"""

    def __init__(self, get_last_q, get_last_q_mod, get_desire_thresholds_row, get_reward_thresholds_action=None, get_last_action=None, get_reward_thresholds_events=None):
        """职责：注入读取最新度量的回调，并初始化运行状态。

        实现步骤：
        1) 保存各读取回调；
        2) 初始化 `_stop` 控制标志；

        函数用来干什么：绑定外部数据提供函数，准备可视化。
        参数说明（输入/输出）：
        - 输入：
          - `get_last_q`（callable->array-like|None）：返回最近一次 Q 值
          - `get_last_q_mod`（callable->array-like|None）：返回最近一次加权后 Q 值
          - `get_desire_thresholds_row`（callable->list[float]）：返回欲望阈值行
          - `get_reward_thresholds_action`（callable(int)->list[float]）：按动作索引返回事件阈值
          - `get_last_action`（callable->int|None）：返回最近一次动作索引
        - 输出：无
        """
        # 步骤 1：保存回调
        self.get_last_q = get_last_q
        self.get_last_q_mod = get_last_q_mod
        self.get_desire_thresholds_row = get_desire_thresholds_row
        self.get_reward_thresholds_action = get_reward_thresholds_action
        self.get_last_action = get_last_action
        self.get_reward_thresholds_events = get_reward_thresholds_events
        # 步骤 2：初始化停止标志
        self._stop = False

    def start(self):
        """职责：启动可视化循环并周期性刷新两个指标图。

        实现步骤：
        1) 检查 matplotlib 可用性；
        2) 开启交互模式并创建两个子图；
        3) 循环：读取最近度量；
        4) 绘制 Q 与 Desire Threshold 双柱图；
        5) 绘制当前动作的事件阈值条形图；
        6) 布局与轻微暂停；

        函数用来干什么：以非阻塞方式周期性更新图像窗口。
        参数说明（输入/输出）：无参数；无返回值。
        """
        # 步骤 1：检查可用性
        if not _HAS_MPL:
            return
        # 步骤 2：开启交互与创建子图
        plt.ion()
        has_ev = self.get_reward_thresholds_events is not None
        has_fb = hasattr(self, 'get_last_events_feedback') and (self.get_last_events_feedback is not None)
        if has_ev and has_fb:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plt.show(block=False)
        while not self._stop:
            # 步骤 3：读取度量
            _time.sleep(0.5)
            last_q = self.get_last_q()
            last_q_mod = self.get_last_q_mod()
            if last_q is None or last_q_mod is None:
                continue
            # 步骤 4：绘制 Q 与 Desire Threshold 双柱图
            ax1.clear()
            n = len(last_q)
            idx = list(range(n))
            width = 0.35
            ax1.bar([i - width/2 for i in idx], last_q, width=width, label='Q')
            thr = self.get_desire_thresholds_row()
            ax1.bar([i + width/2 for i in idx], thr, width=width, label='Desire Threshold')
            ax1.set_title('Per-action Q and Desire Threshold')
            ax1.legend()
            # 步骤 5：绘制事件阈值条形图（事件维度共享）
            ax2.clear()
            if has_ev:
                ev_thr = self.get_reward_thresholds_events()
                ax2.bar(list(range(len(ev_thr))), ev_thr, width=0.6, color='orange')
                ax2.set_title('Event Thresholds (shared)')
                ax2.set_xticks(list(range(len(ev_thr))) if len(ev_thr) > 0 else [])
                ax2.set_ylim(0, max(1.0, max(ev_thr) if len(ev_thr) > 0 else 1.0))
            elif self.get_reward_thresholds_action is not None and self.get_last_action is not None:
                act = int(self.get_last_action() or 0)
                ev_thr = self.get_reward_thresholds_action(act)
                ax2.bar(list(range(len(ev_thr))), ev_thr, width=0.6, color='orange')
                ax2.set_title(f'Action {act} Event Thresholds')
                ax2.set_xticks(list(range(len(ev_thr))) if len(ev_thr) > 0 else [])
                ax2.set_ylim(0, max(1.0, max(ev_thr) if len(ev_thr) > 0 else 1.0))
            if has_ev and has_fb:
                ax3.clear()
                fb_pairs = self.get_last_events_feedback() or []
                ev_ids = [p[0] for p in fb_pairs]
                fb_vals = [p[1] for p in fb_pairs]
                ax3.bar(ev_ids, fb_vals, width=0.6, color='green')
                ax3.set_title('Events Feedback Strengths')
                ax3.set_xticks(ev_ids)
                ax3.set_ylim(0, 1.0)
            # 步骤 6：布局与轻微暂停
            fig.tight_layout()
            plt.pause(0.001)

    def stop(self):
        """职责：停止可视化循环。实现步骤：设置 `_stop` 为 True。

        函数用来干什么：终止循环刷新。
        参数说明（输入/输出）：无参数；无返回值。
        """
        self._stop = True


class InputVisRunner:
    """职责：以固定帧率循环显示输入序列帧，便于观察状态堆叠。

    类用来干什么：展示模型输入的堆叠图像帧，辅助调试与观察输入质量。
    """

    def __init__(self, fps):
        """职责：初始化显示帧率与线程控制状态。

        实现步骤：
        1) 保存帧率；
        2) 初始化停止事件；
        3) 初始化序列缓冲与窗口状态；
        4) 初始化线程句柄；

        函数用来干什么：配置显示器初始状态。
        参数说明（输入/输出）：
        - 输入：`fps`（int|float）：显示帧率
        - 输出：无
        """
        # 步骤 1：保存帧率
        self.fps = fps
        # 步骤 2：初始化停止事件
        self._stop_event = threading.Event()
        # 步骤 3：初始化序列缓冲与窗口状态
        self._seq_np = None
        self._window_initialized = False
        # 步骤 4：初始化线程句柄
        self._thread = None

    def start(self):
        """职责：启动显示线程，循环展示状态帧。

        实现步骤：
        1) 定义显示循环函数；
        2) 在线程中启动显示循环；
        3) 保存线程句柄；

        函数用来干什么：在独立线程中以 `fps` 刷新显示状态堆叠帧。
        参数说明（输入/输出）：无参数；无返回值。
        """
        # 步骤 1：定义显示循环函数
        def display_loop():
            while not self._stop_event.is_set():
                seq = self._seq_np
                if seq is None:
                    time.sleep(0.01)
                    continue
                # 步骤 2：计算帧数并逐帧显示
                if seq.ndim == 4:
                    # 格式: (k, H, W, C)
                    k, H, W, C = seq.shape
                    for i in range(k):
                        if self._stop_event.is_set():
                            break
                        img = seq[i]
                        cv2.imshow("state_t", img)
                        cv2.waitKey(1)
                        if not self._window_initialized:
                            window_utils.set_window_topmost("state_t")
                            window_utils.move_window("state_t", "top_left")
                            self._window_initialized = True
                        time.sleep(1 / self.fps)
                else:
                    # 格式: (Ck, H, W)
                    Ck, H, W = seq.shape
                    k = max(1, Ck // 3)
                    for i in range(k):
                        if self._stop_event.is_set():
                            break
                        frame_chw = seq[3 * i: 3 * (i + 1), :, :]
                        img = np.transpose(frame_chw, (1, 2, 0))
                        cv2.imshow("state_t", img)
                        cv2.waitKey(1)
                        # 步骤 3：首次初始化窗口位置与置顶
                        if not self._window_initialized:
                            window_utils.set_window_topmost("state_t")
                            window_utils.move_window("state_t", "top_left")
                            self._window_initialized = True
                        time.sleep(1 / self.fps)
        # 步骤 2：在线程中启动显示循环
        t = threading.Thread(target=display_loop, daemon=True)
        t.start()
        # 步骤 3：保存线程句柄
        self._thread = t

    def update(self, seq_np):
        """职责：更新即将显示的状态序列。实现步骤：赋值到 `_seq_np`。

        函数用来干什么：替换内部状态序列缓存，供显示线程使用。
        参数说明（输入/输出）：
        - 输入：`seq_np`（np.ndarray，形状 `Ck×H×W`，Ck 为通道堆叠数）
        - 输出：无
        """
        self._seq_np = seq_np

    def stop(self):
        """职责：停止显示线程。实现步骤：设置 `_stop_event` 为触发状态。

        函数用来干什么：安全地结束显示循环。
        参数说明（输入/输出）：无参数；无返回值。
        """
        if self._stop_event:
            self._stop_event.set()