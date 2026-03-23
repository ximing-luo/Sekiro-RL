# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 提供在此仓库中工作的指导。

## 项目概述

**Sekiro-RL** 是一个基于强化学习 (RL) 的《只狼：影逝二度》(Sekiro: Shadows Die Twice) 游戏智能体训练框架。项目采用 **Isaac Lab** 风格架构设计，通过管理器模式 (Manager-based) 实现高度模块化的环境配置与任务定义。

### 核心技术栈

| 类别 | 技术/框架 |
|------|-----------|
| **RL 框架** | Stable-Baselines3 (SB3) |
| **深度学习** | PyTorch |
| **环境接口** | Gymnasium |
| **架构风格** | Isaac Lab (管理器模式) |
| **算法** | PPO (主)、DQN (备用) |

## 架构

代码库遵循受 Isaac Lab 启发的**管理器模式**设计，将环境逻辑分解为独立的管理器 Terms：

- `ObservationTermCfg` – 观测空间定义
- `ActionTermCfg` – 动作空间定义
- `RewardTermCfg` – 奖励函数
- `TerminationTermCfg` – 终止条件
- `EventTermCfg` – 事件触发器

### 关键目录

- `configs/` – 全局配置 (`config.py` 使用数据类分层配置)
- `scripts/reinforcement_learning/` – 训练 (`train.py`) 与推理 (`play.py`) 入口点
- `src/framework/` – RL 框架适配层 (SB3 集成、自定义框架模块)
- `src/gamelab/` – 游戏环境基础设施 (AppLauncher、ManagerBasedRLEnv、管理器、资产、场景)
- `src/model/` – 神经网络模型 (DQN、PPO 特征提取器、通用组件)
- `src/tasks/` – 任务定义层 (只狼任务实现、MDP 组件、注册)
- `outputs/` – 模型输出与日志 (`models/`、`logs/`)

### 设计模式

1. **管理器模式**：对标 Isaac Lab，将环境逻辑分解为独立的管理器 Terms
2. **适配器模式**：`SB3VecEnvAdapter` 将原生 Tensor 环境包装为 SB3 兼容的 `VecEnv` 接口
3. **配置分层**：使用 Python `dataclass` 实现类型安全的分层配置系统，优先级明确

## 命令

### 快速开始

1. 安装核心依赖：
   ```bash
   pip install stable-baselines3 torch gymnasium
   ```
2. 运行基础 PPO 训练：
   ```bash
   python scripts/reinforcement_learning/train.py --task Sekiro-v0 --num_envs 1
   ```
3. 使用 TensorBoard 监控训练：
   ```bash
   tensorboard --logdir logs/sb3/ppo_sekiro
   ```

### 依赖安装

```bash
# 项目使用标准 Python 包管理
pip install stable-baselines3 torch gymnasium
# 其他依赖需根据实际环境补充
```

**额外依赖**（按需安装）：
```bash
pip install opencv-python pywin32 numpy matplotlib tqdm
```

完整 `requirements.txt` 尚未维护；请检查源代码中的导入语句以获取完整列表。

### 训练

```bash
# 基础训练 (PPO)
python scripts/reinforcement_learning/train.py --task Sekiro-v0

# 自定义超参数
python scripts/reinforcement_learning/train.py \
    --task Sekiro-v0 \
    --learning_rate 1e-4 \
    --batch_size 128 \
    --n_steps 4096 \
    --experiment_name my_experiment

# 从断点恢复
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
```

### 主要命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称 | `Sekiro-v0` |
| `--num_envs` | 并行环境数 | `1` |
| `--device` | 计算设备 | `cuda:0` |
| `--headless` | 无头模式 | `False` |
| `--learning_rate` | 学习率覆盖 | 使用 config.py |
| `--batch_size` | 批大小覆盖 | 使用 config.py |
| `--n_steps` | PPO 采集步数 | `2048` |
| `--experiment_name` | 实验名称 | `ppo_sekiro` |
| `--resume` | 从断点恢复 | `False` |
| `--checkpoint` | 模型路径 | `outputs/models/dqn_model.pth` |

### 测试

`tests/` 目录包含以下测试文件（注意该目录被 git 忽略，因此测试为本地专用）：

- `test_sb3_cleaning.py` – 验证 `SekiroCombinedCallback` 是否清理调试目录。
- `test_recorder_manager.py` – 测试 `RecorderManager` 的数据保存逻辑。
- `test_mouse_monitor.py` – 验证鼠标输入监控。

运行所有测试：
```bash
python -m unittest discover tests
```

运行特定测试：
```bash
python -m unittest tests.test_sb3_cleaning
```

### 开发工具

`scripts/tools/` 目录包含多个用于开发和调试的实用脚本：

| 脚本 | 用途 |
|------|------|
| `scripts/tools/check.py` | 验证模型架构与 SB3 训练管线的兼容性 |
| `scripts/tools/stats.py` | 计算模型统计量（参数量、FLOPs、显存占用） |
| `scripts/tools/collect.py` | 启动环境并通过随机动作采集真实观测数据用于基准测试 |
| `scripts/tools/eval.py` | 运行训练模型的评估 |
| `scripts/tools/telemetry.py` | 实时遥测监控 |
| `scripts/tools/view.py` | 可视化录制数据 |

运行方式：`python scripts/tools/<脚本>.py`

### 模仿学习

- **录制**：`python scripts/imitation_learning/record.py` – 通过虚拟摄像头和键盘录制人类游戏过程（仅限 Windows）。按 F9 开始/停止录制，F8 退出。
- **训练**：`python scripts/imitation_learning/train_bc.py` – 从录制的 `.npy` 文件训练行为克隆策略。

## 配置系统

配置遵循 Isaac Lab 风格的分层优先级：

1. **CLI 参数**（最高优先级，显式覆盖）
2. **config.py**（基准配置）
3. **任务特定配置**（如 `SekiroEnvCfg`）

全局配置在 `configs/config.py` 中使用 frozen dataclass 定义。每个任务定义自己的配置类，继承自 `ManagerBasedRLEnvCfg`。

## 开发约定

### 代码风格

- **类型注解**：使用 Python 类型提示 (`typing` 模块)
- **数据类**：配置类统一使用 `@dataclass` 装饰器
- **文档字符串**：模块和类需包含中文文档字符串
- **命名约定**：
  - 配置类后缀 `Cfg` (如 `SekiroEnvCfg`)
  - 管理器 Terms 后缀 `TermCfg` (如 `RewardTermCfg`)

### 架构原则

项目遵循 `.trae/rules/coding.md` 中记载的代码质量哲学方法。关键原则：

1. **逻辑-效能同构**：高效代码应是逻辑本质的自然表达，而非晦涩技巧
2. **逻辑的物理化**：偏好固态、类型化的结构，而非“气态”数据（字典、字符串）
3. **契约的神圣性**：框架应强制执行硬契约，而非在运行时探测或修补（避免在执行路径中使用 `hasattr`、`isinstance` 检查）
4. **维度的正交性**：概念应在维度上完全分离
5. **视觉减法**：最小化缩进层级；使用卫语句和提前返回
6. **职责的纯粹性**：遵循单一职责原则；避免跨层干扰
7. **最少知识原则**：逻辑单元只应与“直接邻居”对话

### 扩展新任务

1. 在 `src/tasks/` 下创建新任务目录
2. 实现继承自 `ManagerBasedRLEnv` 的环境类
3. 实现继承自 `ManagerBasedRLEnvCfg` 的配置类
4. 在 `registration.py` 中注册任务

### 平台注意事项

- 环境设计用于 **Windows**（使用 `win32api`、`win32con`）。
- 项目中的 Shell 命令可能假定 PowerShell 语法（例如，递归删除使用 `-r -fo` 而非 `-rf`）。
- 路径分隔符在 PowerShell 中可正可反，但建议保持一致。

### 关键文件说明

| 文件 | 职责 |
|------|------|
| `configs/config.py` | 全局超参数配置 (学习率、奖励折扣、网络结构等) |
| `src/gamelab/envs/manager_based_env.py` | 基础管理器驱动环境类 (协调仿真和各管理器的生命周期) |
| `src/gamelab/envs/manager_based_rl_env.py` | RL 特定的 `ManagerBasedEnv` 子类 |
| `src/tasks/sekiro/sekiro_env_cfg.py` | 只狼环境具体配置 (观测/奖励/终止 Terms) |
| `src/framework/sb3/adapter.py` | SB3 VecEnv 适配器 (Tensor ↔ NumPy 转换) |
| `src/framework/sb3/runner.py` | 训练循环管理器 (对标 IsaacLab OnPolicyRunner) |
| `src/framework/sb3/ppo_aux.py` | 带辅助损失的 PPO 扩展 (特征相似度损失) |
| `src/model/dqn/stacked.py` | 双流 DQN 网络架构 |
| `src/model/ppo/spatial.py` | PPO 多模态特征提取器 |

## 日志与监控

- **TensorBoard**：训练日志自动记录到 `logs/sb3/{experiment_name}/`
- **模型检查点**：按 `save_freq` 自动保存到 `logs/sb3/{experiment_name}/checkpoints/`
- **配置备份**：训练开始时自动保存 `env.yaml` 和 `agent.yaml`

## Git 说明

- 以下目录被忽略：`__pycache__/`、`.trae`、`logs`、`outputs`、`tests`、`log.md`、`AGENTS.md`、`reward.txt`
- `tests/` 目录被忽略，表明测试为本地专用