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

    def _display_3d_rgb(self, seq):
        """显示 3D RGB 图像 (H, W, C)"""
        # RGB -> BGR for cv2.imshow
        img_bgr = cv2.cvtColor(seq, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("state_t", img_bgr)
        cv2.waitKey(1)
        self._ensure_window_pos()
        time.sleep(1 / self.fps)

    def _display_3d_stacked(self, seq):
        """显示 3D 堆叠图像 (C*T, H, W)"""
        k = max(1, seq.shape[0] // 3)
        for i in range(k):
            if self._stop_event.is_set(): break
            img = np.transpose(seq[3 * i: 3 * (i + 1)], (1, 2, 0))
            # RGB -> BGR for cv2.imshow
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imshow("state_t", img_bgr)
            cv2.waitKey(1)
            self._ensure_window_pos()
            time.sleep(1 / self.fps)

    def _display_4d(self, seq):
        """显示 4D 序列 (B, C, H, W) 或 (T, C, H, W)"""
        for i in range(seq.shape[0]):
            if self._stop_event.is_set(): break
            # RGB -> BGR for cv2.imshow
            img_bgr = cv2.cvtColor(seq[i].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            
            cv2.imshow("state_t", img_bgr)
            cv2.waitKey(1)
            self._ensure_window_pos()
            time.sleep(1 / self.fps)

    def _select_strategy(self, seq):
        """策略路由：根据张量维度选择显示策略。"""
        # 1. 优先检查 3D RGB (HWC 格式)
        if seq.ndim == 3 and seq.shape[-1] == 3:
            return self._display_3d_rgb
        
        # 2. 检查 3D 堆叠 (C*T, H, W 格式)
        if seq.ndim == 3:
            return self._display_3d_stacked
            
        # 3. 检查 4D 序列 (B, C, H, W 或 T, C, H, W)
        if seq.ndim == 4:
            return self._display_4d
            
        return None

    def start(self):
        """职责：启动显示线程，循环展示状态帧。
        
        基于“视觉减法”原则：通过策略分发展平逻辑迷宫。
        """
        def display_loop():
            # 阻塞等待：直到“有数据”或“要停止”
            while self._seq_np is None:
                if self._stop_event.is_set(): 
                    return # 发现停止信号，直接原地解散
                time.sleep(0.01)

            # 策略在启动时确定一次即可
            strategy = self._select_strategy(self._seq_np)
            if not strategy:
                return

            # 主循环：持续展示状态帧，直到“要停止”
            while not self._stop_event.is_set():
                strategy(self._seq_np)

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
