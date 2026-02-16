import time
import threading
import numpy as np
import cv2
import src.gamelab.interfaces.window_utils as window_utils

class InputVisRunner:
    """职责：以固定帧率循环显示输入序列帧，便于观察状态堆叠。
    
    属于 vision 接口的调试辅助工具。
    """

    def __init__(self, fps):
        self.fps = fps
        self._stop_event = threading.Event()
        self._seq_np = None
        self._window_initialized = False
        self._thread = None

    def _display_4d(self, seq):
        """显示 4D 序列 (B, C, H, W) 或 (T, C, H, W)"""
        for i in range(seq.shape[0]):
            if self._stop_event.is_set(): break
            cv2.imshow("state_t", seq[i])
            cv2.waitKey(1)
            self._ensure_window_pos()
            time.sleep(1 / self.fps)

    def _display_3d_rgb(self, seq):
        """显示 3D RGB 图像 (H, W, C)"""
        cv2.imshow("state_t", seq)
        cv2.waitKey(1)
        self._ensure_window_pos()
        time.sleep(1 / self.fps)

    def _display_3d_stacked(self, seq):
        """显示 3D 堆叠图像 (C*T, H, W)"""
        k = max(1, seq.shape[0] // 3)
        for i in range(k):
            if self._stop_event.is_set(): break
            img = np.transpose(seq[3 * i: 3 * (i + 1)], (1, 2, 0))
            cv2.imshow("state_t", img)
            cv2.waitKey(1)
            self._ensure_window_pos()
            time.sleep(1 / self.fps)

    def _select_strategy(self, seq):
        """策略路由：根据张量维度选择显示策略。"""
        if seq.ndim == 4:
            return self._display_4d
        if seq.ndim == 3:
            # 维度内分支：如果最后维度是3，认为是HWC格式，否则认为是堆叠格式
            return self._display_3d_rgb if seq.shape[-1] == 3 else self._display_3d_stacked
        return None

    def start(self):
        """职责：启动显示线程，循环展示状态帧。
        
        基于“视觉减法”原则：通过策略分发展平逻辑迷宫。
        """
        def display_loop():
            while not self._stop_event.is_set():
                seq = self._seq_np
                if seq is None:
                    time.sleep(0.01)
                    continue
                
                strategy = self._select_strategy(seq)
                if strategy:
                    strategy(seq)

        t = threading.Thread(target=display_loop, daemon=True)
        t.start()
        self._thread = t

    def _ensure_window_pos(self):
        """确保窗口位置正确。"""
        if self._window_initialized:
            return
        window_utils.set_window_topmost("state_t")
        window_utils.move_window("state_t", "top_left")
        self._window_initialized = True

    def update(self, seq_np):
        """职责：更新即将显示的状态序列。"""
        self._seq_np = seq_np

    def stop(self):
        """职责：停止显示线程。"""
        if self._stop_event:
            self._stop_event.set()
