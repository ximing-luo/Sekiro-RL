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

def write_csv(log_dir, run_id, step, episode, action, adj_reward, last_q, last_q_mod, raw_reward=None, adv_values=None, adv_values_shrink=None, state_value=None, events=None, noop_1s_count=None, action_recent_counts=None, reward_avg_recent=None, epsilon=None):
    """职责：将训练关键度量以 CSV 形式追加到日志文件。

    实现步骤：
    1) 确保日志目录存在；
    2) 构造写入行的字段；
    3) 判断是否需要写入表头；
    4) 以追加模式写入一行；

    函数用来干什么：将一条训练过程的统计数据（动作、事件、奖励与 Q 值快照）写入 CSV，便于后续分析。
    参数说明（输入/输出）：
    - 输入：
      - `log_dir`（str）：日志目录
      - `run_id`（str）：本次运行标识
      - `step`（int）：训练步编号
      - `action`（int）：选择的动作索引
      - `adj_reward`（float）：调整后的奖励
      - `last_q`（array-like|None）：最近一次原始 Q 值
      - `last_q_mod`（array-like|None）：最近一次加权后的 Q 值
    - 输出：无（写文件副作用）
    """
    # 步骤 1：确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'train_metrics.csv')
    # 步骤 2：构造数据行字段
    events_str = ';'.join([f"{int(e)}" for e in (events or [])])
    row = {
        'run_id': run_id,
        'step': int(step),
        'episode': int(episode),
        'action': int(action),
        'adj_reward': float(adj_reward),
        'raw_reward': (float(raw_reward) if raw_reward is not None else None),
        'reward_avg_recent': (float(reward_avg_recent) if reward_avg_recent is not None else None),
        'q_values': ';'.join([f"{float(x):.6f}" for x in (list(last_q) if last_q is not None else [])]),
        'q_values_mod': ';'.join([f"{float(x):.6f}" for x in (list(last_q_mod) if last_q_mod is not None else [])]),
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

def write_json(log_dir, run_id, step, episode, action, adj_reward, last_q, last_q_mod, fps=None, adv_values=None, state_value=None, events=None, raw_reward=None, noop_1s_count=None, adv_values_shrink=None, action_recent_counts=None, reward_avg_recent=None, epsilon=None):
    """职责：将最近一次训练度量写入 JSON 文件，便于前端读取与可视化。

    实现步骤：
    1) 确保日志目录存在；
    2) 整理并转换各字段类型；
    3) 组装 payload 数据结构；
    4) 写入 `latest.json`；

    函数用来干什么：以结构化 JSON 的形式输出最新一次训练状态，供 Streamlit/前端或其他工具读取。
    参数说明（输入/输出）：
    - 输入：参数同 `write_csv`。
    - 输出：无（写文件副作用）
    """
    # 步骤 1：确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    # 步骤 2：整理并转换各字段类型
    q_vals = [float(x) for x in (list(last_q) if last_q is not None else [])]
    q_mods = [float(x) for x in (list(last_q_mod) if last_q_mod is not None else [])]
    adv_vals = [float(x) for x in (list(adv_values) if adv_values is not None else [])]
    adv_vals_shrink = [float(x) for x in (list(adv_values_shrink) if adv_values_shrink is not None else [])]
    # 步骤 3：组装 payload
    payload = {
        'run_id': run_id,
        'step': int(step),
        'episode': int(episode),
        'action': int(action),
        'events': [int(e) for e in (list(events) if events is not None else [])],
        'adj_reward': float(adj_reward),
        'raw_reward': (float(raw_reward) if raw_reward is not None else None),
        'reward_avg_recent': (float(reward_avg_recent) if reward_avg_recent is not None else None),
        'q_values': q_vals,
        'q_values_mod': q_mods,
        'adv_values': adv_vals,
        'adv_values_shrink': adv_vals_shrink,
        'state_value': (float(state_value) if state_value is not None else None),
        'fps': (float(fps) if fps is not None else None),
        'noop_1s_count': (int(noop_1s_count) if noop_1s_count is not None else None),
        'action_recent_counts': [int(x) for x in (list(action_recent_counts) if action_recent_counts is not None else [])],
        'epsilon': (float(epsilon) if epsilon is not None else None),
    }
    # 步骤 4：写入文件
    path = os.path.join(log_dir, 'latest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

def load_last_training_stats(log_dir: str):
    """职责：从日志目录中加载最后的训练步数和回合数。

    实现步骤：
    1) 尝试从 latest.json 加载；
    2) 尝试从 train_metrics.csv 加载（如果 CSV 中的步数更大）；
    3) 返回 (last_step, last_episode)；

    函数用来干什么：恢复训练进度，避免从零开始。
    参数说明：
    - 输入：`log_dir`（str）：日志目录
    - 输出：`(last_step, last_episode)`（tuple）
    """
    last_step = 0
    last_episode = 0

    # 1. 尝试从 latest.json 加载
    json_path = os.path.join(log_dir, 'latest.json')
    if os.path.isfile(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            last_step = int(data.get('step', 0))
            last_episode = int(data.get('episode', 0))
        except Exception:
            pass

    # 2. 尝试从 train_metrics.csv 加载更准确的/最新的数据
    csv_path = os.path.join(log_dir, 'train_metrics.csv')
    if os.path.isfile(csv_path):
        try:
            import pandas as pd
            # 读取 CSV 并获取最后一行
            df = pd.read_csv(csv_path)
            if not df.empty:
                last_row = df.iloc[-1]
                csv_step = int(last_row.get('step', 0))
                csv_episode = int(last_row.get('episode', 0))
                # 如果 CSV 的步数更新，则以 CSV 为准
                if csv_step > last_step:
                    last_step = csv_step
                    last_episode = csv_episode
        except Exception:
            pass

    return last_step, last_episode

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