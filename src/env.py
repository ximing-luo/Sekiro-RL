"""
模块用途：游戏环境封装，负责动作执行、状态采集、奖励计算与经验写入。

包含：
- 类：Sekiro
- 方法：take_action, step, reset, pause_game, get_reward 等

边界：
- 负责：桥接采集模块、图像指标提取、奖励计算与回放缓冲写入
- 不负责：训练优化、阈值机制内部实现、可视化与日志写入
"""
import threading
import time
import date.image_process as image_process
from date.replay_buffer import ReplayBuffer
from date.capture import FrameCapture
from date.reward import design_event_rewards
from date.actions_map import get_action_callable, ACTION_LABELS, assert_config_consistency, action_count
from date.metrics import extract_metrics
from utils.getkeys import key_check
from utils import window_utils
import config


class Sekiro(object):
    def __init__(self, observation_w, observation_h, action_dim=None, pos = "offscreen", debug_vis_fps=0, capture_fps=60, n_step_rewards: int = 20):
        super().__init__()

        # 基础观测尺寸
        self.observation_dim = observation_w * observation_h
        self.width = observation_w
        self.height = observation_h

        # 动作空间配置
        self.action_dim = int(action_dim) if action_dim is not None else int(action_count())
        assert_config_consistency(self.action_dim)
        if len(ACTION_LABELS) != self.action_dim:
            print(f"动作标签数量({len(ACTION_LABELS)})与动作维度({self.action_dim})不一致")

        # 指标提取窗口参数
        self.blood_window = config.BLOOD_WINDOW
        self.stamina_window = config.STAMINA_WINDOW

        # 环境状态变量
        self.boss_blood = 0
        self.self_blood = 0
        self.boss_stamina = 0
        self.self_stamina = 0
        self.emergence_break = 0
        self.obs = None
        self.expected_num_events = 9
        self.over = False
        self.n_step_rewards = int(max(1, n_step_rewards))
        
        # 回放缓冲与图像采集模块
        self.replay_buffer = ReplayBuffer(size=50, frame_history_len=config.FRAME_HISTORY_LEN)
        self.capture_fps = capture_fps
        self._frame_capture = FrameCapture(
            camera_index=config.CAMERA_INDEX,
            camera_width=config.CAMERA_WIDTH,
            camera_height=config.CAMERA_HEIGHT,
            fps=self.capture_fps,
            replay_buffer=self.replay_buffer,
            target_width=self.width,
            target_height=self.height,
        )
        self._frame_capture.start() # 启动图像采集线程

        # 调试可视化
        if debug_vis_fps and debug_vis_fps > 0:
            self._start_debug_visualization(debug_vis_fps) # 启动调试可视化线程

        # 窗口移动与激活
        window_utils.move_window("Sekiro", pos, True) #offscreen，center
        # print(f"窗口移动至：{pos}")
        window_utils.activate_window_by_title_contains("Sekiro", True)

        # 计时器与计数器
        self.last_time = time.time() # 上次环境交互时间戳
        self._step_counter = 0 # 环境交互步数计数器

        # 状态记录
        self.last_events_feedback = []
        self.last_events = []
        self.last_total_reward_raw = 0.0
        self.last_noop_count_1s = 0

    # 移除了奖励阈值和兴奋阈值相关方法

    def _start_image_capture(self):
        """启动图像采集线程（封装调用）。"""
        try:
            if hasattr(self, "_frame_capture"):
                self._frame_capture.start()
        except Exception:
            pass

    # 调试：可视化当前输入
    def _start_debug_visualization(self, fps):
        '''
        启动输入状态调试可视化线程。

        数据来源：由训练循环在 `train.py` 中调用 `env.update_debug_visual_input(stacked_np)`
        传入的状态帧序列（形状为 CHW 堆叠的多帧）。

        fps 含义：可视化刷新帧率（每秒显示帧数），用于控制窗口更新速度。

        实现内容：实例化并启动 `date.visualization.InputVisRunner`，在线程中将
        堆叠帧逐张转换为 HWC 图像后通过 OpenCV 窗口展示，窗口置顶并定位。
        '''
        from date.visualization import InputVisRunner
        self._input_vis_runner = InputVisRunner(fps)
        self._input_vis_runner.start() # 启动调试可视化线程
    # 更新调试窗口输入序列（CHW 堆叠帧）
    def update_debug_visual_input(self, seq_np):
        self._input_vis_runner.update(seq_np)  
    # 停止调试窗口线程函数
    def stop_debug_visualization(self):
        if hasattr(self, "_input_vis_runner") and self._input_vis_runner:
            self._input_vis_runner.stop()  

    #数据处理函数引用
    def self_blood_count(self, self_bgr):
        self_blood = image_process.self_blood_count(self_bgr)
        return self_blood
    def boss_blood_count(self, boss_bgr):
        boss_blood = image_process.boss_blood_count(boss_bgr)
        return boss_blood
    def self_stamina_count(self, self_bgr):
        self_stamina = image_process.self_stamina_count(self_bgr)
        return self_stamina
    def boss_stamina_count(self, boss_bgr):
        boss_stamina = image_process.boss_stamina_count(boss_bgr)
        return boss_stamina
    
    #动作设计
    def take_action(self, action):
        window_utils.activate_window_by_title_contains("Sekiro")
        fn = get_action_callable(action)
        fn()
    #反馈设计  
    def _calculate_loss_reward(self, action_id):
        return calculate_loss_reward(action_id)
        
    # 事件判别：根据前后状态与动作，判定发生的事件，结束标志与紧急中断计数
    def detect_events(
        self,
        boss_blood, next_boss_blood, self_blood, next_self_blood,
        boss_stamina, next_boss_stamina, self_stamina, next_self_stamina,
        emergence_break,
    ):
        events = []
        done = 0
        # [0] 自身死亡
        if self_blood < 100 and next_self_blood - self_blood > 500:
            print("\033[33mself dead\033[0m")
            if emergence_break < 1:
                done = 1
                emergence_break += 1
                events = [0]
                return events, done, emergence_break
            done = 1
            emergence_break = 100
            events = [0]
            return events, done, emergence_break
        # [1] Boss死亡
        if next_boss_blood == 0 and boss_blood - next_boss_blood > 50 and next_boss_stamina > 400:
            print("\033[32mboss dead\033[0m")
            if emergence_break < 1:
                done = 1
                emergence_break += 1
                events = [1]
                return events, done, emergence_break
            done = 1
            emergence_break = 100
            events = [1]
            return events, done, emergence_break
        # 其他事件判别（与原 compute_reward 的事件列表一致）
        # [2] 自身掉血
        if next_self_blood - self_blood < -2:
            events.append(2)
        # [3] 自身回血
        if 100 <= self_blood < 300 and next_self_blood - self_blood >= 100:
            events.append(3)
        # [4] 自身血量过低
        if next_self_blood <= 200:
            events.append(4)
        # [5] Boss掉血
        if next_boss_blood - boss_blood <= -5:
            events.append(5)
        # [6] 自身架势上升
        if next_self_stamina - self_stamina >= 2:
            events.append(6)
        # [7] Boss架势上升
        if next_boss_stamina - boss_stamina >= 2:
            events.append(7)
        # [8] Boss架势过低
        if next_boss_stamina <= 20:
            events.append(8)
        return events, done, emergence_break

    # 奖励计算：根据事件与状态设计每事件奖励分量与总奖励
    def get_reward(
        self,
        boss_blood, next_boss_blood, self_blood, next_self_blood,
        boss_stamina, next_boss_stamina, self_stamina, next_self_stamina,
        action,
        self_blood_gamma=0.6, boss_blood_gamma=0.5, self_stamina_gamma=0.5, boss_stamina_gamma=0.5
    ):
        events, done, emergence_break = self.detect_events(
            boss_blood, next_boss_blood, self_blood, next_self_blood,
            boss_stamina, next_boss_stamina, self_stamina, next_self_stamina,
            self.emergence_break,
        )
        total_reward, components = design_event_rewards(
            boss_blood, next_boss_blood, self_blood, next_self_blood,
            boss_stamina, next_boss_stamina, self_stamina, next_self_stamina,
            action,
            events,
            self_blood_gamma=self_blood_gamma, boss_blood_gamma=boss_blood_gamma, self_stamina_gamma=self_stamina_gamma, boss_stamina_gamma=boss_stamina_gamma,
        )
        return total_reward, done, emergence_break, events, components

    
    # 环境交互（动作执行->数据收集->奖励计算）
    def step(self, action):
        # 动作执行（异步线程，避免主循环阻塞）
        threading.Thread(target=self.take_action, args=(action,), daemon=True).start()

        # 数据收集
        observe = self._frame_capture.latest_frame
        next_self_blood, next_boss_blood, next_self_stamina, next_boss_stamina = extract_metrics(
            observe, self.blood_window, self.stamina_window
        )

        # 根据环境反馈计算奖励
        reward, done, emergence_break, events, components = self.get_reward(
            self.boss_blood, next_boss_blood, self.self_blood, next_self_blood,
            self.boss_stamina, next_boss_stamina, self.self_stamina, next_self_stamina,
            action,
            self_blood_gamma=0.8, boss_blood_gamma=1, self_stamina_gamma=0.8, boss_stamina_gamma=2)
        # 记录原始奖励与事件
        self.last_total_reward_raw = float(reward)
        self.last_events = list(events)
        try:
            self.last_events_feedback = [[int(e), float(components.get(int(e), 0.0))] for e in events]
        except Exception:
            self.last_events_feedback = []
        self.self_blood = next_self_blood
        self.boss_blood = next_boss_blood
        self.self_stamina = next_self_stamina
        self.boss_stamina = next_boss_stamina
        self.emergence_break = emergence_break
        
        # 直接使用原始奖励，不再进行阈值调整
        adjusted_reward = float(reward)

        self.replay_buffer.store_effect(action, adjusted_reward, done)

        # 环境数据打印
        self._step_counter += 1
        if self._step_counter % 10 == 0:
            print("狼的血量:",next_self_blood,"boos的血量:",next_boss_blood)
            print("狼的架势条:",next_self_stamina,"boos的架势条:",next_boss_stamina,"\n")
        print(self.last_time - time.time())
        self.last_time = time.time()
        return adjusted_reward
        
    #环境交互（暂停游戏）
    def pause_game(self, paused):
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('start game')
            else:
                paused = True
                print('pause game')
            time.sleep(1)  # 添加短暂延迟以防止单次按键多次触发

        # if self.emergence_break == 100:
        #     paused = True
        #     self.emergence_break = 0
        #     print('emergence break')

        if paused:
            print('paused')
            while paused:  # 循环直到游戏不再暂停
                keys = key_check()
                if 'T' in keys:
                    paused = False  # 解除暂停
                    print('start game')
                    time.sleep(1)  # 解除暂停后延迟
                if 'P' in keys:
                    paused = False
                    self.over = True
                    time.sleep(1)  # 添加短暂延迟以防止单次按键多次触发
        return paused

    #环境重置（初始化）
    def reset(self):
        # 重新抓取原始窗口截图以更新血量/精力基线
        observe = self._frame_capture.latest_frame
        sb, bb, ss, bs = extract_metrics(observe, self.blood_window, self.stamina_window)
        self.self_blood = sb
        self.boss_blood = bb
        self.self_stamina = ss
        self.boss_stamina = bs

if __name__ == '__main__':
    pos = 'offscreen'
    window_utils.move_window("Sekiro", pos) #offscreen，center



    




    
