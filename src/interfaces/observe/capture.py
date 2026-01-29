"""
模块用途：摄像头采集与帧入缓冲的后台管理。

包含：
- 类：FrameCapture（start/stop）

边界：
- 负责：采集、缩放、写入 ReplayBuffer，并提供 latest_frame
- 不负责：奖励计算、环境逻辑、训练流程
"""
import threading
import time
import ctypes
import cv2
import numpy as np

class FrameCapture:
    def __init__(self, camera_index, camera_width, camera_height, fps, target_width, target_height):
        self.camera_index = camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fps = fps
        self.target_width = target_width
        self.target_height = target_height
        self.cap = None
        self.latest_frame = None
        self._stop = False

    def start(self):
        def capture_loop():
            # 避免系统DPI缩放模糊
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
            # 采集OBS虚拟摄像头数据
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            ret, frame = self.cap.read()
            if not ret:
                return
            period = 1.0 / float(self.fps) if self.fps and self.fps > 0 else 0.0
            while not self._stop:
                t0 = time.time()
                ret, observe = self.cap.read()
                if not ret or observe is None:
                    continue

                # 核心代码逻辑：读取帧、缩放、更新最新帧
                observe_resize = cv2.resize(observe, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
                self.latest_frame = np.array(observe_resize).reshape(self.target_height, self.target_width, 3)

                if period > 0.0:
                    dt = time.time() - t0
                    sleep_t = period - dt
                    if sleep_t > 0:
                        time.sleep(sleep_t)
        t = threading.Thread(target=capture_loop, daemon=True)
        t.start()

    def stop(self):
        self._stop = True
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

if __name__ == '__main__':
    # 调试代码：实例化 FrameCapture 并显示画面
    # 默认使用配置中的参数，如果没有配置则手动指定
    try:
        from configs.config import cfg
        camera_idx = cfg.scene.camera_index
        c_width = cfg.scene.camera_width
        c_height = cfg.scene.camera_height
        fps = cfg.scene.capture_fps
        t_width = cfg.scene.img_width
        t_height = cfg.scene.img_height
    except Exception as e:
        print(f"配置加载失败，使用默认参数: {e}")
        camera_idx = 1
        c_width = 1920
        c_height = 1080
        fps = 60
        t_width = 480//2
        t_height = 270//2

    capture = FrameCapture(
        camera_index=camera_idx,
        camera_width=c_width,
        camera_height=c_height,
        fps=fps,
        target_width=t_width,
        target_height=t_height
    )
    capture.start()
    
    print(f"开始预览 (Camera Index: {camera_idx})，按 'q' 键退出...")
    try:
        while True:
            if capture.latest_frame is not None:
                # 显示采集到的画面
                cv2.imshow('FrameCapture Debug', capture.latest_frame)
            
            # 等待 1ms 检查按键，按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n用户中断调试")
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print("已停止采集并关闭窗口")

