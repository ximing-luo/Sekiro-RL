from __future__ import annotations
import time
import torch
from typing import Optional, Any, Dict, Sequence
from .manager_based_env_cfg import ManagerBasedEnvCfg
from src.gamelab.scene.interactive_scene import InteractiveScene
from src.gamelab.sim import SimulationContext, SimulationCfg
from src.gamelab.sim.sensors.vision_sensor import VisionSensor
from src.gamelab.interfaces import window_utils
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
        # 1. 固化契约：仅接受已就绪的配置对象
        self.cfg = cfg
        self.render_mode = render_mode
        
        self.scene: Optional[InteractiveScene] = None
        
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

    @property
    def num_envs(self) -> int:
        """环境中的并行实例数量。"""
        return self.cfg.scene.num_envs

    @property
    def device(self) -> str:
        """环境运行的计算设备。"""
        return self.cfg.device

    def _setup_managers(self):
        """初始化仿真器与各管理器。"""
        scene_cfg = self.cfg.scene
        
        # A. 创建场景 (物理属性归位：从 self.device 获取)
        self.scene = InteractiveScene(scene_cfg, device=self.device)
        
        # B. 初始化仿真上下文 (对标 Isaac Lab 的 sim)
        sim_cfg = SimulationCfg(
            dt=1.0 / scene_cfg.capture_fps,
            window_title="Sekiro",
            window_pos=scene_cfg.pos,
            device=self.device,
            headless=self.cfg.headless,
            debug_vis_fps=scene_cfg.debug_vis_fps
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
          
        # F. 实例化逻辑管理器 (传递配置和环境实例)
        self.action_manager = ActionManager(self.cfg.actions, self)
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        self.event_manager = EventManager(self.cfg.events, self)
        self.command_manager = CommandManager(self.cfg.commands, self)
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculums, self)

    def reset(self, env_ids: Sequence[int] | None = None) -> Dict[str, torch.Tensor]:
        """重置环境。"""
        env_ids = env_ids if env_ids is not None else list(range(self.num_envs))
            
        self.scene.reset(env_ids)
        self.action_manager.reset(env_ids)
        self.reward_manager.reset(env_ids)
        self.termination_manager.reset(env_ids)
        self.event_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        self.curriculum_manager.reset(env_ids)
        
        return self.observation_manager.step()

    def step(self, action: torch.Tensor) -> tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """执行一个环境步。
        
        实现执行必然性：移除试探性判断，确立线性数据流。
        """
        # 0. 状态采样
        self.scene.write_data_to_sim()

        # 1. 物理步进
        self.action_manager.step(action)
        self.sim.step()
        self.scene.update(self.sim.cfg.dt)
        
        # 2. 逻辑采样
        self.event_manager.step(mode="step")

        # 3. 结果解算
        reward, reward_components = self.reward_manager.step(self.sim.cfg.dt)
        done, time_out = self.termination_manager.step()
        obs = self.observation_manager.step()
        
        self.common_step_counter += 1
        
        # 封装 info
        info = {
            "time_out": time_out,
            "step": self.common_step_counter,
            "events": self.event_manager.recent_events,
            "reward_components": reward_components
        }
        
        return obs, reward, done, time_out, info

    def activate_window(self):
        window_utils.activate_window_by_title(self.sim.cfg.window_title)

    def close(self):
        """清理资源。"""
        self.sim.stop()
