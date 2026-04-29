# Ashina RL 框架使用指南

## 概述

Ashina 是一个**原生并行**的轻量离线策略 RL 框架，核心分三层：

```
接口层         VectorEnv           ← 向量化环境接口，框架只规定不实现
                                        真正并行由用户环境负责
学习层         DQNAlgorithm        ← Double DQN
               DQNPolicy           ← 推理 + ε-greedy
数据层         Collector           ← N 环境并行采集 + 自动存储
               ReplayBuffer        ← 环形经验回放
调度层         OffPolicyTrainer    ← 迭代式训练：先采集再反复学习
便利层         DQNAgent            ← 开箱即用高层封装
工具层         GymnasiumWrapper    ← Gymnasium 向量环境包装器
```

### 设计理念

```
框架只规定接口（VectorEnv），不关心具体实现。
环境可以是：
  · GPU 向量化（IsaacLab 风格，一步 8096 个环境）
  · CPU 多进程（SubprocVectorEnv）
  · Gymnasium SyncVectorEnv / AsyncVectorEnv
环境提供者负责实现真正的 reset() 和 step()，返回 batch 数据。
```

---

## 快速开始

### 安装依赖

```bash
pip install torch numpy gymnasium
```

### 核心类快速概览

| 类 | 职责 | 并行方式 |
|----|------|----------|
| `VectorEnv` | 向量化环境接口 | 框架不实现，用户负责 |
| `GymnasiumWrapper` | Gymnasium 向量环境包装器 | 处理 torch tensor → numpy |
| `Collector` | N 环境并行采集 | 采集并行（N 个 step 一起走） |
| `DQNAlgorithm` | Double DQN 学习 | 单 batch 大样本学习（batch_size=2048） |
| `OffPolicyTrainer` | 调度采集+学习 | 迭代式（采集→反复学习） |
| `DQNAgent` | 高层封装 | 开箱即用 |

---

## 用法一：DQNAgent 高层接口

```python
import gymnasium as gym
from framework.ashina.highlevel.dqn_agent import DQNAgent, DQNConfig
from framework.ashina.data.buffer import ReplayBuffer
from framework.ashina.env.gymnasium_wrapper import GymnasiumWrapper

# 用 GymnasiumWrapper 包装原生 SyncVectorEnv
env_fns = [lambda: gym.make("CartPole-v1") for _ in range(1024)]
vec_env = GymnasiumWrapper(gym.vector.SyncVectorEnv(env_fns))

agent = DQNAgent(
    vec_env=vec_env,
    action_dim=2,
    buffer=ReplayBuffer(size=100000),
    params=DQNConfig(lr=1e-3, batch_size=2048),
)

# 迭代式训练
result = agent.learn(steps_per_iter=24, learn_epochs=4)
# → {"rew": ..., "batches_per_epoch": 12, "epochs": 4}

# 单步推理
state = vec_env.reset()[0]
action = agent.act(state)

# 保存 / 加载
agent.save("model.pth")
agent.load("model.pth")
```

### DQNConfig 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | `1e-3` | Adam 学习率 |
| `gamma` | `0.99` | 折扣因子 |
| `target_update_freq` | `1000` | 目标网络同步间隔（优化步数） |
| `epsilon_start` | `1.0` | 初始探索率 |
| `epsilon_end` | `0.01` | 最小探索率 |
| `epsilon_decay` | `0.995` | 探索率乘法衰减系数 |
| `batch_size` | `2048` | 每次梯度更新的样本量 |

---

## 用法二：低层手动组装

需要自定义网络结构或训练逻辑时使用：

```python
import gymnasium as gym
from framework.ashina.algorithm.modelfree.dqn import DQNAlgorithm
from framework.ashina.algorithm.net.discrete import QNetwork
from framework.ashina.data.buffer import ReplayBuffer
from framework.ashina.data.collector import Collector
from framework.ashina.trainer.offpolicy import OffPolicyTrainer
from framework.ashina.env.gymnasium_wrapper import GymnasiumWrapper

env_fns = [lambda: gym.make("CartPole-v1") for _ in range(1024)]
vec_env = GymnasiumWrapper(gym.vector.SyncVectorEnv(env_fns))

# 自定义网络
model = QNetwork(input_dim=4, action_dim=2, hidden_dims=[256, 256])

# 组装算法
algo = DQNAlgorithm(action_dim=2, model=model, lr=1e-3)

# 组装数据流（N 环境并行采集）
buffer = ReplayBuffer(size=100000)
collector = Collector(algo, vec_env, buffer)

# 调度层
trainer = OffPolicyTrainer(algo, collector, batch_size=2048)

# 迭代式训练
result = trainer.train_iteration(steps_per_iter=24, learn_epochs=4)
# → {"rew": ..., "batches_per_epoch": 12, "epochs": 4}
```

### 传入自定义网络

`DQNAlgorithm` 只要求 `model` 输出形状为 `(B, action_dim)`，内部结构自由：

```python
import torch.nn as nn

class MyCNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 32, 8, stride=4)
        self.fc = nn.Linear(32 * 9 * 9, 2)
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))

algo = DQNAlgorithm(action_dim=2, model=MyCNNNet())
```

---

## 训练逻辑

框架采用**迭代式训练**：每次迭代先大规模采集数据，再反复学习。

```
总采集量 = num_envs × steps_per_iter = 1024×24 = 24576 条/迭代

for iteration in range(num_iterations):
    │
    ├── 采集阶段
    │   for _ in range(steps_per_iter):
    │       actions = algo(obs)
    │       env.step(actions) → buffer.add(×1024)
    │   ← buffer 新增 24576 条经验
    │
    └── 学习阶段
        algo.pre_collect()  ← epsilon 衰减（每迭代一次）
        for epoch in range(learn_epochs):           ← 4 轮
            for _ in range(batches_per_epoch):       ← 12 batch/轮
                batch = buffer.sample(batch_size)    ← 采 2048 条
                algo.learn(batch)

每迭代总更新量: 4×12 = 48 次
```

### 批次量推导

```
batch_size = 2048           (每次梯度更新的样本量，由用户定义)
batches_per_epoch = 24576 / 2048 = 12  (由总采集量 / batch_size 自然得出)
```

---

## 目录结构

```
src/framework/ashina/
├── algorithm/
│   ├── base.py              Policy / Algorithm 抽象基类
│   ├── net/
│   │   ├── common.py        MLP 基础块
│   │   └── discrete.py      QNetwork（离散动作空间）
│   └── modelfree/
│       └── dqn.py           DQNPolicy + DQNAlgorithm
├── data/
│   ├── batch.py             Batch 数据容器
│   ├── buffer.py            环形经验回放
│   └── collector.py         向量化采集器（N 环境并行）
├── env/
│   ├── venvs.py             VectorEnv 接口（只定义不实现）
│   └── gymnasium_wrapper.py Gymnasium 包装器
├── trainer/
│   └── offpolicy.py         OffPolicyTrainer（迭代式调度）
├── highlevel/
│   └── dqn_agent.py         DQNAgent 高层封装
├── utils/
│   └── tensor_utils.py      to_numpy / to_tensor 工具
└── exploration/
    └── random.py            GaussianNoise / OUNoise
```
