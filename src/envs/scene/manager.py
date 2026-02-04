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
    def __init__(self, observation_w, observation_h, pos="top_right", capture_fps=60):
        self.width = observation_w
        self.height = observation_h
        self.capture_fps = capture_fps
        self.pos = pos
        self.window_title = "Sekiro"
        
        # 图像采集模块
        self._frame_capture = FrameCapture(
            camera_index=config.cfg.scene.camera_index,
            camera_width=config.cfg.scene.camera_width,
            camera_height=config.cfg.scene.camera_height,
            fps=self.capture_fps,
            target_width=self.width,
            target_height=self.height,
        )
        
        # 调试可视化
        self._input_vis_runner = None

    def get_latest_frame(self):
        """获取最新的采集帧。"""
        return self._frame_capture.latest_frame

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
        self._frame_capture.stop()

    def __getstate__(self):
        """序列化保护：排除不可序列化的可视化线程对象。"""
        state = self.__dict__.copy()
        # 可视化运行器包含线程，无法被 pickle
        state['_input_vis_runner'] = None
        return state

    def __setstate__(self, state):
        """反序列化：恢复状态。"""
        self.__dict__.update(state)
        self._input_vis_runner = None

if __name__ == '__main__':
    # 调试代码：直接运行此类测试采集与可视化
    import os
    import sys
    # 确保能够导入 src
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from configs.config import cfg
    
    # 1. 实例化场景管理器
    manager = SceneManager(
        observation_w=cfg.scene.img_width,
        observation_h=cfg.scene.img_height,
        pos="top_left",
        capture_fps=cfg.scene.capture_fps
    )
    
    # 2. 启动采集
    print("正在初始化场景管理器...")
    manager.setup()
    
    # 3. 开启调试可视化 (假设配置中 debug_vis_fps > 0)
    vis_fps = cfg.ui.debug_vis_fps if cfg.ui.debug_vis_fps > 0 else 30
    print(f"开启调试可视化，FPS: {vis_fps}")
    manager.start_debug_visualization(vis_fps)
    
    print("开始循环测试，按 Ctrl+C 退出...")
    try:
        while True:
            frame = manager.get_latest_frame()
            if frame is not None:
                # 更新可视化数据
                manager.update_debug_visualization(frame)
                # 打印一些基本信息
                if time.time() % 2 < 0.05: # 每2秒打印一次
                    print(f"当前帧形状: {frame.shape}, 数据范围: [{frame.min()}, {frame.max()}]")
            else:
                print("等待帧采集...")
            
            time.sleep(1 / vis_fps)
    except KeyboardInterrupt:
        print("\n调试被用户中断")
    finally:
        manager.stop()
        print("场景管理器已停止")
