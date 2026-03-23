# Sekiro-RL

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-333333.svg)](https://github.com/DLR-RM/stable-baselines3)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Latest-000000.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **只狼：影逝二度** 强化学习智能体训练框架 —— 基于 Isaac Lab 架构风格，采用管理器模式实现高度模块化的 RL 环境

---

## 📖 目录

- [简介](#-简介)
- [特性](#-特性)
- [架构设计](#-架构设计)
- [快速开始](#-快速开始)
- [使用方法](#-使用方法)
- [项目结构](#-项目结构)
- [配置说明](#-配置说明)
- [开发指南](#-开发指南)
- [许可证](#-许可证)

---

## 🎮 简介

**Sekiro-RL** 是一个专为《只狼：影逝二度》设计的强化学习训练框架，旨在训练能够自主掌握游戏机制的智能体。项目借鉴了 **NVIDIA Isaac Lab** 的设计理念，采用管理器模式 (Manager-based Pattern) 实现高度解耦的环境配置与任务定义。

### 核心技术栈

| 组件 | 技术选型 |
|------|----------|
| **RL 框架** | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) |
| **深度学习** | [PyTorch](https://pytorch.org/) |
| **环境接口** | [Gymnasium](https://gymnasium.farama.org/) |
| **架构风格** | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (管理器模式) |
| **主算法** | PPO / DQN |

---

## ✨ 特性

- **🏗️ 管理器模式架构** —— 将环境逻辑解耦为独立的管理器 Terms（观测/动作/奖励/终止/事件）
- **🔧 高度可配置** —— 基于 Python `dataclass` 的分层配置系统，支持 CLI 动态覆盖
- **🧪 模块化设计** —— 任务、环境、模型、训练器完全解耦，易于扩展
- **📊 TensorBoard 集成** —— 自动记录训练指标与超参数备份
- **🚀 性能优化** —— 支持 CUDA 加速、多环境并行采样、TF32 矩阵运算
- **🎯 适配器模式** —— `SB3VecEnvAdapter` 无缝对接 Stable-Baselines3

---

## 🏛️ 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  train.py   │  │   play.py   │  │      cli_args.py        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Framework Adapter Layer                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  SB3OnPolicyRunner  │  SB3VecEnvAdapter  │  PPO Auxiliary │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                       Environment Layer                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              ManagerBasedRLEnv (Base Class)               │  │
│  │  ┌─────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────┐ │  │
│  │  │Observ.  │ │ Action │ │ Reward │ │Terminate │ │Event │ │  │
│  │  │ Manager │ │Manager │ │Manager │ │ Manager  │ │Manager│ │  │
│  │  └─────────┘ └────────┘ └────────┘ └──────────┘ └──────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        Task Layer                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │   SekiroEnv  │  SekiroEnvCfg  │  mdp/ (Terms Definitions) │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure Layer                       │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ AppLauncher │ │ Scene Manager│ │   Assets    │ │  Utils   │ │
│  └─────────────┘ └──────────────┘ └─────────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于 GPU 加速)
- 《只狼：影逝二度》游戏环境

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/your-username/Sekiro-RL.git
cd Sekiro-RL

# 安装核心依赖
pip install stable-baselines3 torch gymnasium

# 其他依赖根据实际环境补充
```

### 验证安装

```bash
# 运行环境测试
python src/tasks/sekiro/env.py
```

---

## 📋 使用方法

### 训练智能体

```bash
# 基础训练 (PPO 算法)
python scripts/reinforcement_learning/train.py --task Sekiro-v0

# 自定义超参数
python scripts/reinforcement_learning/train.py \
    --task Sekiro-v0 \
    --learning_rate 1e-4 \
    --batch_size 128 \
    --n_steps 4096 \
    --experiment_name my_experiment

# 多环境并行训练
python scripts/reinforcement_learning/train.py \
    --task Sekiro-v0 \
    --num_envs 8 \
    --device cuda:0

# 从断点恢复训练
python scripts/reinforcement_learning/train.py \
    --task Sekiro-v0 \
    --resume \
    --checkpoint logs/sb3/ppo_sekiro/2024-01-01_12-00-00/checkpoints/sekiro_ppo_latest.zip
```

### 推理/演示

```bash
# 运行训练好的智能体
python scripts/reinforcement_learning/play.py \
    --task Sekiro-v0 \
    --checkpoint path/to/model.zip

# 无头模式（无 GUI）
python scripts/reinforcement_learning/play.py \
    --task Sekiro-v0 \
    --checkpoint path/to/model.zip \
    --headless
```

### 主要命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称 | `Sekiro-v0` |
| `--num_envs` | 并行环境数 | `1` |
| `--device` | 计算设备 | `cuda:0` |
| `--headless` | 无头模式 | `False` |
| `--learning_rate` | 学习率覆盖 | 使用 `config.py` |
| `--batch_size` | 批大小覆盖 | 使用 `config.py` |
| `--n_steps` | PPO 采集步数 | `2048` |
| `--experiment_name` | 实验名称 | `ppo_sekiro` |
| `--resume` | 从断点恢复 | `False` |
| `--checkpoint` | 模型权重路径 | `outputs/models/` |

---

## 📁 项目结构

```
Sekiro-RL/
├── configs/                  # 全局配置
│   └── config.py             # 数据类分层配置 (Scene/RL/Train 等)
├── scripts/                  # 执行脚本入口
│   ├── reinforcement_learning/
│   │   ├── train.py          # 训练入口 (PPO)
│   │   ├── play.py           # 推理/演示入口
│   │   └── cli_args.py       # 命令行参数定义
│   ├── imitation_learning/   # 模仿学习脚本 (待扩展)
│   └── tools/                # 工具脚本
├── src/                      # 核心源代码
│   ├── framework/            # RL 框架适配层
│   │   ├── sb3/              # SB3 集成
│   │   │   ├── runner.py     # 训练循环管理器
│   │   │   ├── adapter.py    # VecEnv 适配器
│   │   │   └── ppo_aux.py    # 带辅助损失的 PPO
│   │   └── ashina/           # 自定义框架模块
│   ├── gamelab/              # 游戏环境基础设施
│   │   ├── app/              # 应用启动器 (AppLauncher)
│   │   ├── envs/             # 管理器基类环境
│   │   ├── interfaces/       # 交互接口（按键模拟、图像抓取、内存遥测）
│   │   ├── managers/         # 管理器实现 (观测/动作/奖励等)
│   │   ├── assets/           # 游戏资产配置
│   │   └── scene/            # 场景管理
│   ├── model/                # 神经网络模型
│   │   ├── dqn/              # DQN 模型 (双流架构)
│   │   ├── ppo/              # PPO 特征提取器
│   │   └── components/       # 通用组件
│   ├── tasks/                # 任务定义层
│   │   ├── sekiro/           # 只狼任务实现
│   │   │   ├── env.py        # Sekiro 环境类
│   │   │   ├── sekiro_env_cfg.py  # 环境配置
│   │   │   └── mdp/          # MDP 组件 (观测/奖励/终止/事件)
│   │   └── registration.py   # 任务注册中心
│   └── utils/                # 工具函数
├── tests/                    # 单元测试
├── outputs/                  # 模型输出与日志
│   ├── models/               # 保存的模型权重
│   └── logs/                 # TensorBoard 日志
├── README.md                 # 本文件
└── LICENSE                   # 开源许可证
```

---

## ⚙️ 配置说明

### 配置优先级

遵循 Isaac Lab 风格的分层优先级：

1. **CLI 参数** (最高优先级，显式覆盖)
2. **`configs/config.py`** (基准配置)
3. **任务特定配置** (如 `SekiroEnvCfg`)

### 核心配置类

| 配置类 | 职责 |
|--------|------|
| `SceneConfig` | 基础环境与采集配置 (分辨率/FPS/摄像头) |
| `TrainConfig` | 训练超参数 (学习率/折扣因子/PPO 参数) |
| `EpsilonConfig` | 探索策略参数 (Epsilon Greedy) |
| `PERConfig` | 优先经验回放 (PER) 参数 |
| `PathConfig` | 路径与日志配置 |

### 配置示例

```python
# configs/config.py
@dataclass(frozen=True)
class TrainConfig:
    """训练超参数 (PPO/SB3)。"""
    learning_rate: float = 2e-5
    gamma: float = 0.91
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 5
    ent_coef: float = 0.01
    clip_range: float = 0.3
```

---

## 🛠️ 开发指南

### 扩展新任务

1. 在 `src/tasks/` 下创建新任务目录
2. 继承 `ManagerBasedRLEnv` 实现环境类
3. 继承 `ManagerBasedRLEnvCfg` 实现配置类
4. 在 `registration.py` 中注册任务

```python
# src/tasks/registration.py
task_registry.register("MyTask-v0", MyTaskEnv, MyTaskEnvCfg)
```

### 代码风格

- **类型注解**：使用 Python 类型提示 (`typing` 模块)
- **数据类**：配置类统一使用 `@dataclass` 装饰器
- **文档字符串**：模块和类需包含中文文档字符串
- **命名约定**：
  - 配置类后缀 `Cfg` (如 `SekiroEnvCfg`)
  - 管理器 Terms 后缀 `TermCfg` (如 `RewardTermCfg`)

### 测试与验证

```bash
# 环境快速验证
python src/tasks/sekiro/env.py

# 配置验证（修改 config.py 后重新运行训练）
python scripts/reinforcement_learning/train.py --task Sekiro-v0
```

### 日志与监控

- **TensorBoard**：训练日志自动记录到 `logs/sb3/{experiment_name}/`
- **模型检查点**：按 `save_freq` 自动保存到 `logs/sb3/{experiment_name}/checkpoints/`
- **配置备份**：训练开始时自动保存 `env.yaml` 和 `agent.yaml`

启动 TensorBoard 查看训练进度：

```bash
tensorboard --logdir logs/sb3/
```

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。

---

## 🙏 致谢

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 可靠的 RL 算法库
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - 架构设计灵感来源
- [FromSoftware](https://www.fromsoftware.jp/) - 《只狼：影逝二度》

---

<div align="center">

**Sekiro-RL** —— 只狼不死，修罗未止 🐺⚔️

[返回顶部](#sekiro-rl)

</div>
