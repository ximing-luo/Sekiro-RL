import time
import threading
import numpy as np
import cv2
from src.gamelab.interfaces.system import window as window_utils

class InputVisRunner:
    """职责：以固定帧率循环显示输入序列帧，便于观察状态堆叠。"""

    def __init__(self, fps):
        self.fps = fps
        self._stop_event = threading.Event()
        self._seq_np = None
        self._window_initialized = False
        self._thread = None

    def start(self):
        """职责：启动显示线程，循环展示状态帧。"""
        def display_loop():
            while not self._stop_event.is_set():
                seq = self._seq_np
                if seq is None:
                    time.sleep(0.01)
                    continue
                if seq.ndim == 4:
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
                elif seq.ndim == 3:
                    if seq.shape[-1] == 3:
                        img = seq
                        cv2.imshow("state_t", img)
                        cv2.waitKey(1)
                        if not self._window_initialized:
                            window_utils.set_window_topmost("state_t")
                            window_utils.move_window("state_t", "top_left")
                            self._window_initialized = True
                        time.sleep(1 / self.fps)
                    else:
                        Ck, H, W = seq.shape
                        k = max(1, Ck // 3)
                        for i in range(k):
                            if self._stop_event.is_set():
                                break
                            frame_chw = seq[3 * i: 3 * (i + 1), :, :]
                            img = np.transpose(frame_chw, (1, 2, 0))
                            cv2.imshow("state_t", img)
                            cv2.waitKey(1)
                            if not self._window_initialized:
                                window_utils.set_window_topmost("state_t")
                                window_utils.move_window("state_t", "top_left")
                                self._window_initialized = True
                            time.sleep(1 / self.fps)
        t = threading.Thread(target=display_loop, daemon=True)
        t.start()
        self._thread = t

    def update(self, seq_np):
        """职责：更新即将显示的状态序列。"""
        self._seq_np = seq_np

    def stop(self):
        """职责：停止显示线程。"""
        if self._stop_event:
            self._stop_event.set()
