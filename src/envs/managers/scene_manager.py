import time
import src.interfaces.system.window as window_utils
from src.interfaces.observe.capture import FrameCapture
from src.interfaces.system.input import key_check
import configs.config as config

class SceneManager:
    """
    场景管理器：负责窗口控制、图像采集线程管理以及游戏暂停/恢复逻辑。
    对应 Isaac Lab 中的 Scene 概念，但针对 Sekiro 进行了适配。
    """
    def __init__(self, observation_w, observation_h, replay_buffer, pos="offscreen", capture_fps=60):
        self.width = observation_w
        self.height = observation_h
        self.replay_buffer = replay_buffer
        self.capture_fps = capture_fps
        self.pos = pos
        self.window_title = "Sekiro"
        
        # 图像采集模块
        self._frame_capture = FrameCapture(
            camera_index=config.CAMERA_INDEX,
            camera_width=config.CAMERA_WIDTH,
            camera_height=config.CAMERA_HEIGHT,
            fps=self.capture_fps,
            replay_buffer=self.replay_buffer,
            target_width=self.width,
            target_height=self.height,
        )
        
        # 调试可视化
        self._input_vis_runner = None

    def setup(self):
        """初始化场景：启动采集、移动并激活窗口。"""
        self._frame_capture.start()
        window_utils.move_window(self.window_title, self.pos, True)
        window_utils.activate_window_by_title_contains(self.window_title, True)

    def activate_window(self):
        """确保窗口处于激活状态。"""
        window_utils.activate_window_by_title_contains(self.window_title)

    def start_debug_visualization(self, fps):
        """启动调试可视化线程。"""
        from src.visualization.logger import InputVisRunner
        self._input_vis_runner = InputVisRunner(fps)
        self._input_vis_runner.start()

    def update_debug_visualization(self, seq_np):
        """更新调试可视化数据。"""
        if self._input_vis_runner:
            self._input_vis_runner.update(seq_np)

    def stop_debug_visualization(self):
        """停止调试可视化。"""
        if self._input_vis_runner:
            self._input_vis_runner.stop()
            self._input_vis_runner = None

    def check_pause(self, paused):
        """检查并处理游戏暂停/恢复逻辑。"""
        keys = key_check()
        if 'T' in keys:
            paused = not paused
            print('start game' if not paused else 'pause game')
            time.sleep(1)

        if paused:
            print('paused')
            while paused:
                keys = key_check()
                if 'T' in keys:
                    paused = False
                    print('start game')
                    time.sleep(1)
                if 'P' in keys:
                    paused = False
                    # 这里可能需要通知 Env 结束
                    time.sleep(1)
                    return True, paused # (should_over, paused)
        return False, paused

    def stop(self):
        """停止所有资源。"""
        self.stop_debug_visualization()
        # 注意：FrameCapture 目前可能没有显示的 stop 方法，
        # 如果有的话应该在这里调用。
