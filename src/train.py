import os
import sys
import time

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import threading
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import csv
import json
from torch.utils.tensorboard import SummaryWriter
import torchvision
from src.tasks.sekiro.env import Sekiro
from src.tasks.sekiro.sekiro_env_cfg import SekiroEnvCfg
from src.policies.dqn_agent import DQNAgent
from src.envs.mdp.actions import action_count
from src.interfaces.system.input import key_check
import src.interfaces.system.window as window_utils
import configs.config as config

def _init_env_agent(pos, img_width, img_height, action_dim, model_path, n_step_rewards):
    ad = action_dim if action_dim is not None else int(action_count())
    
    # 使用新的配置驱动初始化
    cfg = SekiroEnvCfg()
    cfg.scene.pos = pos
    cfg.scene.observation_w = img_width
    cfg.scene.observation_h = img_height
    cfg.scene.debug_vis_fps = 60
    cfg.n_step_rewards = n_step_rewards
    
    env = Sekiro(cfg=cfg)
    agent = DQNAgent(img_width, img_height, ad, buffer=env.replay_buffer, model_file=model_path, n_step_rewards=n_step_rewards)
    if os.path.isfile(model_path):
        try:
            agent.load_model()
            print(f"已从现有模型恢复：{model_path}")
        except Exception as e:
            print(f"加载已有模型失败（忽略继续训练）：{e}")
    return env, agent

def _wait_buffer(env):
    print("等待缓冲区填充初始帧...")
    while env.replay_buffer.video_num_in_buffer < env.replay_buffer.frame_history_len:
        time.sleep(0.05)
    time.sleep(0.2)  # 额外等待0.2秒，确保暂停开始存入正确游戏画面

def _register_tensorboard_hooks(agent: DQNAgent, writer: SummaryWriter):
    """注册 Hook 以在 TensorBoard 中记录特征图。"""
    def get_activation(name):
        def hook(model, input, output):
            # 仅在特定步数记录（通过 agent.current_step 控制）
            # 假设每 1000 步记录一次
            step = getattr(agent, 'current_step', 0)
            if step > 0 and step % 1000 == 0:
                try:
                    # output shape: (Batch, Channel, H, W)
                    # 取第一个样本: (Channel, H, W)
                    img = output[0].detach().cpu()
                    
                    # 归一化到 [0, 1]
                    mn, mx = img.min(), img.max()
                    if mx - mn > 1e-9:
                        img = (img - mn) / (mx - mn)
                    else:
                        img = torch.zeros_like(img)

                    # 如果通道数过多，只取前 64 个通道进行展示
                    if img.shape[0] > 64:
                        img = img[:64]
                    
                    # 将通道维度作为 batch 维度，制作网格: (C, 1, H, W) -> Grid
                    grid = torchvision.utils.make_grid(img.unsqueeze(1), nrow=8, padding=2, normalize=False)
                    writer.add_image(f'Features/{name}', grid, step)
                except Exception as e:
                    print(f"Hook error for {name}: {e}")
        return hook

    model = agent.algorithm.eval_net
    # 动态发现所有 Conv2d 层以支持不同架构的模型（如 ResNet, SimpleDQN 等）
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    
    layers_to_hook = {}
    if len(conv_layers) > 0:
        # 均匀采样最多 5 个卷积层进行可视化
        indices = np.linspace(0, len(conv_layers) - 1, min(5, len(conv_layers)), dtype=int)
        for i, idx in enumerate(indices):
            name, layer = conv_layers[idx]
            # 缩短显示名称，只保留最后一部分
            display_name = name.split('.')[-1] if '.' in name else name
            layers_to_hook[f"{i}_{display_name}"] = layer

    for name, layer in layers_to_hook.items():
        layer.register_forward_hook(get_activation(name))
    if layers_to_hook:
        print(f"已自动注册 {len(layers_to_hook)} 个 TensorBoard Hooks。")
    else:
        print("未发现可注册 Hook 的卷积层。")

def _load_last_training_counters(log_dir: str):
    last_step = 0
    last_episode = 0
    json_path = os.path.join(log_dir, 'latest.json')
    if os.path.isfile(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                j = json.load(f)
            if isinstance(j, dict):
                last_step = int(j.get('step', 0) or 0)
                ep = j.get('episode', None)
                if ep is not None:
                    try:
                        last_episode = int(ep)
                    except Exception:
                        pass
        except Exception:
            pass
    csv_path = os.path.join(log_dir, 'train_metrics.csv')
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, 'r', newline='') as f:
                r = csv.DictReader(f)
                prev_step = None
                episodes = 0
                last_row_step = None
                last_row_episode = None
                for row in r:
                    try:
                        s = int(row.get('step', 0) or 0)
                    except Exception:
                        s = 0
                    last_row_step = s
                    if 'episode' in row and row['episode'] not in (None, '', 'None'):
                        try:
                            last_row_episode = int(row['episode'])
                        except Exception:
                            pass
                    if prev_step is None:
                        prev_step = s
                        episodes = 1 if s > 0 else 0
                    else:
                        if s < prev_step:
                            episodes += 1
                        prev_step = s
                if isinstance(last_row_step, int) and last_row_step > last_step:
                    last_step = last_row_step
                if last_row_episode is not None:
                    last_episode = last_row_episode
                elif episodes > 0:
                    last_episode = episodes
        except Exception:
            pass
    return last_step, last_episode

def _select_action_and_store(env: Sekiro, agent: DQNAgent, step: int, save_interval: int = 100, epsilon: float = 0.0) -> int:
    k = env.replay_buffer.frame_history_len  # 需要堆叠的历史帧数，用于构造状态
    stacked_np = env.replay_buffer.get_latest_observation(k)  # 从缓冲区取最近k帧并拼接为观测
    env.update_debug_visual_input(stacked_np)  # 更新调试可视化输入，便于观察模型状态
    # 直接传递 numpy 数组，代理内部会处理转换
    action = agent.select_action(stacked_np, epsilon=epsilon)  # 由代理根据当前状态选择动作（含探索策略）
    env.replay_buffer.store_latest_observation(stacked_np)  # 将本次观测写入经验缓冲区供训练使用
    return action  # 返回本步选中的动作索引

# （streamlit）记录训练指标
def _log_metrics(agent: DQNAgent, env: Sekiro, step, episode, action, reward, recent_rewards_deque, fps=None, epsilon=None):
    env.log_manager.log_step(agent, env, step, episode, action, reward, recent_rewards_deque, fps, epsilon)

def _maybe_optimize(agent, env, step):
    freq = getattr(config, 'OPTIMIZE_EVERY_STEPS', 1)
    try:
        freq = int(freq)
    except Exception:
        freq = 1
    if freq <= 0:
        freq = 1
    if step % freq == 0:
        threading.Thread(target=agent.learn, daemon=True).start()

def _maybe_print(step, recent_rewards, last_print_time):
    if time.time() - last_print_time > 1.0:
        avg = (sum(recent_rewards) / len(recent_rewards)) if len(recent_rewards) > 0 else 0.0
        print(f"\033[91mstep={step} avg_reward_10={avg:.3f}\033[0m")
        return time.time()
    return last_print_time

def _maybe_save(agent: DQNAgent, last_save_steps, flag = False):
    if agent.optimize_count - last_save_steps >= 100 or flag:
        agent.save_model()
        print(f"模型已保存：{agent.model_file}")
        return agent.optimize_count
    return last_save_steps

def train_agent(
    pos="offscreen",
    img_width=480,
    img_height=270,
    action_dim=None,
    total_interaction_steps=10000,
    model_path=config.MODEL_PATH,
    n_step_rewards: int = 20,
):
    '''
    训练代理模型。

    参数:
        pos (str): 游戏窗口位置，"offscreen" 表示无窗口。
        img_width (int): 游戏图像宽度。
        img_height (int): 游戏图像高度。
        action_dim (int, optional): 动作维度。如果为 None，则从环境中获取。
        total_interaction_steps (int): 总交互步数。
        model_path (str): 模型保存路径。
    '''
    # 初始化环境与代理模型
    env, agent = _init_env_agent(pos, img_width, img_height, action_dim, model_path, n_step_rewards)
    
    # 确保目录存在
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    
    # 注册 TensorBoard Hook
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    _register_tensorboard_hooks(agent, writer)
    
    log_dir = config.LOG_DIR
    last_step, last_episode = _load_last_training_counters(log_dir)
    global_step = int(last_step)
    env.pause_game(True) # 暂停，可以慢慢把游戏画面调好
    _wait_buffer(env) # 等待环境的视频缓冲区加载满，确保初始状态稳定
    for episode in range(int(last_episode) + 1, int(last_episode) + 1 + 10):
        env.reset() # 重置参数，将游戏状态设为初始状态
        # 初始化主训练参数
        last_save_steps = 0
        last_print_time = time.time()
        recent_rewards = deque(maxlen=10)
        
        # Epsilon 探索参数
        EPS_START = 1.0
        EPS_END = 0.1
        EPS_DECAY = 50000 # 探索衰减步数

        for _ in range(total_interaction_steps):
            # 帧数控制：限制循环频率最高为60Hz，确保与游戏同步
            _loop_start = time.time()
            # 核心代码逻辑缩进，外层为帧数控制

            # 计算 Epsilon
            epsilon = EPS_END + (EPS_START - EPS_END) * max(0, (EPS_DECAY - global_step) / EPS_DECAY)

            # 动作与环境交互：选动作、环境步进并进行奖励调整
            global_step += 1
            agent.current_step = global_step
            action = _select_action_and_store(env, agent, global_step, 500, epsilon=epsilon)
            # 在环境中执行一步动作并根据阈值模块调整奖励并存入经验缓冲区
            reward = env.step(action)

            # 训练优化：基于回放缓冲区进行异步优化
            _maybe_optimize(agent, env, global_step)

            # 日志与统计：累计奖励、写日志并打印近期均值
            recent_rewards.append(float(reward))
            events_feedback = getattr(env, "last_events_feedback", [])
            # FPS 计算与打印
            _dt = time.time() - _loop_start
            _fps = (1.0 / _dt) if _dt > 1e-6 else 0.0
            print(f"FPS={_fps:.2f} EPS={epsilon:.3f}")
            _log_metrics(agent, env, global_step, episode, action, reward, recent_rewards, fps=_fps, epsilon=epsilon)
            last_print_time = _maybe_print(global_step, recent_rewards, last_print_time)

            # 模型保存：按优化步计数定期保存模型
            last_save_steps = _maybe_save(agent, last_save_steps)

            # 游戏控制：用户按键操作（T 键暂停/继续，P结束训练并保存模型，boos两次死亡后暂停）
            env.pause_game(False)
            if env.over:
                break
            if "P" in key_check():
                env.over = True
                break

            
            _dt = time.time() - _loop_start
            _target = 1.0 / 60.0
            if _dt < _target:
                time.sleep(_target - _dt)
                
        _maybe_save(agent, last_save_steps, True)
        print(f"第{episode}轮训练结束，模型已保存。")
        print(f"模型已保存：{agent.model_file}")
        if env.over:
            break
    print("训练结束。")


def play_agent(
    pos="offscreen",
    img_width=480,
    img_height=270,
    action_dim=None,
    total_interaction_steps=100000,
    model_path=config.MODEL_PATH,
    n_step_rewards: int = 20,
):
    '''
    运行代理模型（推理模式，不训练，不探索）。

    参数:
        pos (str): 游戏窗口位置，"offscreen" 表示无窗口。
        img_width (int): 游戏图像宽度。
        img_height (int): 游戏图像高度。
        action_dim (int, optional): 动作维度。如果为 None，则从环境中获取。
        total_interaction_steps (int): 总交互步数。
        model_path (str): 模型保存路径。
    '''
    # 初始化环境与代理模型
    env, agent = _init_env_agent(pos, img_width, img_height, action_dim, model_path, n_step_rewards)
    
    print(f"正在以推理模式运行模型: {model_path}")
    
    global_step = 0
    env.pause_game(True) # 暂停，可以慢慢把游戏画面调好
    _wait_buffer(env) # 等待环境的视频缓冲区加载满，确保初始状态稳定
    
    env.reset() # 重置参数
    
    last_print_time = time.time()
    recent_rewards = deque(maxlen=10)
    
    # 推理模式：不探索
    epsilon = 0.0
    
    # 确保模型处于评估模式
    if hasattr(agent.algorithm.eval_net, 'eval'):
        agent.algorithm.eval_net.eval()

    for _ in range(total_interaction_steps):
        # 帧数控制
        _loop_start = time.time()

        global_step += 1
        agent.current_step = global_step
        
        # 选择动作（epsilon=0）并存入 buffer（为了获取下一帧的历史堆叠）
        # 注意：虽然我们不训练，但 select_action_and_store 内部依赖 buffer 来构建 state
        action = _select_action_and_store(env, agent, global_step, 500, epsilon=epsilon)
        
        # 环境步进
        reward = env.step(action)

        # 不调用 _maybe_optimize (不训练)

        # 统计
        recent_rewards.append(float(reward))
        
        # FPS 计算与打印
        _dt = time.time() - _loop_start
        _fps = (1.0 / _dt) if _dt > 1e-6 else 0.0
        print(f"FPS={_fps:.2f} Reward={reward:.3f}")
        
        # 可选：记录日志，但通常推理模式不需要污染训练日志
        # events_feedback = getattr(env, "last_events_feedback", [])
        # _log_metrics(...) 
        
        last_print_time = _maybe_print(global_step, recent_rewards, last_print_time)

        # 不保存模型

        # 游戏控制
        env.pause_game(False)
        if env.over:
            print("游戏结束")
            break
        if "P" in key_check():
            env.over = True
            print("用户停止运行")
            break
        
        _dt = time.time() - _loop_start
        _target = 1.0 / 60.0
        if _dt < _target:
            time.sleep(_target - _dt)
            
    print("运行结束。")


if __name__ == "__main__":
    train_agent()
    # play_agent() # 默认改为运行推理模式，或者让用户自己选择
    time.sleep(1.0)
    window_utils.move_window("Sekiro", "center")