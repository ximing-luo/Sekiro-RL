from __future__ import annotations
import time
import torch
from typing import Optional, Any, Dict, Sequence
from .manager_based_env_cfg import ManagerBasedEnvCfg
from src.gamelab.scene.interactive_scene import InteractiveScene
from src.gamelab.sim import SimulationContext, SimulationCfg
from src.gamelab.sim.sensors.vision_sensor import VisionSensor
from src.gamelab.managers import (
    ActionManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
    EventManager,
    CommandManager,
    RecorderManager,
    CurriculumManager
)
import configs.config as config

class ManagerBasedEnv:
    """基础管理器驱动环境类。
    
    负责协调仿真桥接器 (Sim) 和各个管理器的生命周期。
    对标 Isaac Lab 的 ManagerBasedEnv。
    """
    def __init__(self, cfg: ManagerBasedEnvCfg, render_mode: str | None = None):
        self.cfg = cfg
        self.render_mode = render_mode
        
        # 基础属性 (对标 Isaac Lab)
        self.num_envs = 1  # 目前单机单实例
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 初始化场景与资产
        self.scene: Optional[InteractiveScene] = None
        
        # 2. 初始化仿真桥接器 (物理底座)
        
        # 2. 初始化逻辑管理器
        self.action_manager: Optional[ActionManager] = None
        self.observation_manager: Optional[ObservationManager] = None
        self.reward_manager: Optional[RewardManager] = None
        self.termination_manager: Optional[TerminationManager] = None
        self.event_manager: Optional[EventManager] = None
        self.command_manager: Optional[CommandManager] = None
        self.recorder_manager: Optional[RecorderManager] = None
        self.curriculum_manager: Optional[CurriculumManager] = None

        # 记录环境步数
        self.common_step_counter = 0

    def _setup_managers(self):
        """初始化仿真器与各管理器。"""
        scene_cfg = self.cfg.scene
        self.num_envs = scene_cfg.num_envs
        
        # A. 创建场景
        self.scene = InteractiveScene(scene_cfg, device=self.device)
        
        # B. 初始化仿真上下文 (对标 Isaac Lab 的 sim)
        sim_cfg = SimulationCfg(
            dt=1.0 / scene_cfg.capture_fps,
            window_title="Sekiro",
            window_pos=scene_cfg.pos,
            headless=False # 目前强制开启 GUI 以便观测
        )
        self.sim = SimulationContext(sim_cfg)
        
        # C. 挂载传感器
        vision_sensor = VisionSensor(
            camera_index=config.cfg.scene.camera_index,
            camera_width=config.cfg.scene.camera_width,
            camera_height=config.cfg.scene.camera_height,
            fps=scene_cfg.capture_fps,
            target_width=scene_cfg.observation_w,
            target_height=scene_cfg.observation_h
        )
        self.sim.add_sensor("vision", vision_sensor)
        
        # D. 启动仿真
        self.sim.setup()
        
        # E. 实例化逻辑管理器 (传递配置和环境实例)
        self.action_manager = ActionManager(self.cfg.actions, self)
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        self.event_manager = EventManager(self.cfg.events, self)
        self.command_manager = CommandManager(self.cfg.commands, self)
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculums, self)
        
        # 调试可视化 (可选)
        if scene_cfg.debug_vis_fps > 0:
            from src.gamelab.interfaces.capture.visualizer import InputVisRunner
            self._input_vis_runner = InputVisRunner(scene_cfg.debug_vis_fps)
            self._input_vis_runner.start()

    def reset(self, env_ids: Sequence[int] | None = None) -> Dict[str, torch.Tensor]:
        """重置环境。"""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
            
        # 重置场景资产
        self.scene.reset(env_ids)
            
        # 重置各管理器
        self.action_manager.reset(env_ids)
        self.reward_manager.reset(env_ids)
        self.termination_manager.reset(env_ids)
        self.event_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        self.curriculum_manager.reset(env_ids)
        
        # 获取初始观测
        obs = self.observation_manager.compute_observations()
        return obs

    def step(self, action: torch.Tensor) -> tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """执行一个环境步。"""
        # 1. 执行动作
        self.action_manager.apply_action(action)
        
        # 2. 同步仿真状态
        self.sim.step()
        
        # 3. 同步游戏状态 (从内存读取资产数据)
        self.scene.update(self.sim.cfg.dt)
        
        # 3. 计算奖励、终止和事件
        reward = self.reward_manager.compute_reward()
        done, time_out = self.termination_manager.compute_terminations()
        
        # 4. 获取最新观测
        obs = self.observation_manager.compute_observations()
        
        # 5. 更新步数计数器
        self.common_step_counter += 1
        
        # 封装 info
        info = {
            "time_out": time_out,
            "step": self.common_step_counter
        }
        
        return obs, reward, done, time_out, info

    def activate_window(self):
        if self.sim:
            window_utils.activate_window_by_title_contains(self.sim.cfg.window_title)

    def close(self):
        """清理资源。"""
        if self.sim:
            self.sim.stop()
        if hasattr(self, '_input_vis_runner') and self._input_vis_runner:
            self._input_vis_runner.stop()
