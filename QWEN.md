# Sekiro-RL 项目上下文

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

### 项目架构

```
Sekiro-RL/
├── configs/              # 全局配置 (config.py - 数据类分层配置)
├── scripts/              # 执行脚本入口
│   └── reinforcement_learning/
│       ├── train.py      # 训练入口 (PPO)
│       ├── play.py       # 推理/演示入口
│       └── cli_args.py   # 命令行参数定义
├── src/                  # 核心源代码
│   ├── framework/        # RL 框架适配层
│   │   ├── sb3/          # SB3 集成 (Runner, Adapter, PPO 扩展)
│   │   └── ashina/       # 自定义框架模块
│   ├── gamelab/          # 游戏环境基础设施
│   │   ├── app/          # 应用启动器 (AppLauncher)
│   │   ├── envs/         # 管理器基类环境 (ManagerBasedRLEnv)
│   │   ├── managers/     # 管理器实现 (观测/动作/奖励等)
│   │   ├── assets/       # 游戏资产配置
│   │   └── scene/        # 场景管理
│   ├── model/            # 神经网络模型
│   │   ├── dqn/          # DQN 模型 (SimpleDQN, 双流架构)
│   │   ├── ppo/          # PPO 特征提取器 (ResNet, EfficientNet)
│   │   └── components/   # 通用组件
│   ├── tasks/            # 任务定义层
│   │   ├── sekiro/       # 只狼任务实现
│   │   │   ├── env.py    # Sekiro 环境类
│   │   │   ├── sekiro_env_cfg.py  # 环境配置
│   │   │   └── mdp/      # MDP 组件 (观测/奖励/终止/事件)
│   │   └── registration.py  # 任务注册中心
│   └── utils/            # 工具函数
└── outputs/              # 模型输出与日志
    ├── models/           # 保存的模型权重
    └── logs/             # TensorBoard 日志
```

### 设计模式

1. **管理器模式 (Manager-based)**：对标 Isaac Lab，将环境逻辑分解为独立的管理器 Terms：
   - `ObservationTermCfg` - 观测空间定义
   - `ActionTermCfg` - 动作空间定义
   - `RewardTermCfg` - 奖励函数
   - `TerminationTermCfg` - 终止条件
   - `EventTermCfg` - 事件触发器

2. **适配器模式**：`SB3VecEnvAdapter` 将原生 Tensor 环境包装为 SB3 兼容的 `VecEnv` 接口。

3. **配置分层**：使用 Python `dataclass` 实现类型安全的分层配置系统。

## 构建与运行

### 依赖安装

```bash
# 项目使用标准 Python 包管理
pip install stable-baselines3 torch gymnasium
# 其他依赖需根据实际环境补充
```

### 训练命令

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

### 推理/演示命令

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

## 开发约定

### 配置优先级

遵循 Isaac Lab 风格的分层优先级：
1. **CLI 参数** (最高优先级，显式覆盖)
2. **config.py** (基准配置)
3. **任务特定配置** (如 `SekiroEnvCfg`)

### 代码风格

- **类型注解**：使用 Python 类型提示 (`typing` 模块)
- **数据类**：配置类统一使用 `@dataclass` 装饰器
- **文档字符串**：模块和类需包含中文文档字符串
- **命名约定**：
  - 配置类后缀 `Cfg` (如 `SekiroEnvCfg`)
  - 管理器 Terms 后缀 `TermCfg` (如 `RewardTermCfg`)

### 测试实践

- 环境测试：运行 `env.py` 的 `__main__` 块进行快速验证
- 配置验证：修改 `configs/config.py` 后需重新运行训练脚本验证

### 扩展新任务

1. 在 `src/tasks/` 下创建新任务目录
2. 继承 `ManagerBasedRLEnv` 实现环境类
3. 继承 `ManagerBasedRLEnvCfg` 实现配置类
4. 在 `registration.py` 中注册任务

### 关键文件说明

| 文件 | 职责 |
|------|------|
| `configs/config.py` | 全局超参数配置 (学习率、奖励折扣、网络结构等) |
| `src/tasks/sekiro/sekiro_env_cfg.py` | 只狼环境的具体配置 (观测/奖励/终止 Terms) |
| `src/framework/sb3/adapter.py` | SB3 VecEnv 适配器 (Tensor ↔ NumPy 转换) |
| `src/framework/sb3/runner.py` | 训练循环管理器 (对标 IsaacLab OnPolicyRunner) |
| `src/framework/sb3/ppo_aux.py` | 带辅助损失的 PPO 扩展 (特征相似度损失) |
| `src/model/dqn/stacked.py` | 双流 DQN 网络架构 |
| `src/model/ppro/spatial.py` | PPO 多模态特征提取器 |

### 日志与监控

- **TensorBoard**：训练日志自动记录到 `logs/sb3/{experiment_name}/`
- **模型检查点**：按 `save_freq` 自动保存到 `logs/sb3/{experiment_name}/checkpoints/`
- **配置备份**：训练开始时自动保存 `env.yaml` 和 `agent.yaml`
