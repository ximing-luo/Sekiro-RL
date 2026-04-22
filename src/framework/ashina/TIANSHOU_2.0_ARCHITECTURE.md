# Tianshou 2.0 架构设计文档 (Architecture Guide)

本文档旨在梳理 Tianshou 2.0 的核心架构与目录职责，帮助开发者理解其从“以策略为中心”向“以算法为中心”的演进，以及高阶 API 的引入逻辑。

---

## 📂 核心目录结构概览

### 1. [algorithm/](file:///d:/Axon/ANN/tianshou/tianshou/algorithm) —— 框架的“大脑” (The Brain)
这是 2.0 版本最重要的重构点，取代了旧版的 `policy` 目录。它将“行动准则”与“学习逻辑”彻底解耦。

- **[algorithm_base.py](file:///d:/Axon/ANN/tianshou/tianshou/algorithm/algorithm_base.py)**: 定义了 `Policy` (数据平面) 和 `Algorithm` (控制平面) 基类。
- **modelfree/**: 无模型算法实现（PPO, DQN, SAC 等）。
- **modelbased/**: 基于模型的算法（ICM, PSRL 等）。
- **imitation/**: 模仿学习与离线 RL（GAIL, BCQ, CQL 等）。
- **multiagent/**: 多智能体算法支持。
- **optim.py**: 统一的优化器工厂管理。

### 2. [highlevel/](file:///d:/Axon/ANN/tianshou/tianshou/highlevel) —— 实验的“管家” (The Manager)
2.0 引入的声明式配置层，旨在消除样板代码，提高实验可复现性。

- **[experiment.py](file:///d:/Axon/ANN/tianshou/tianshou/highlevel/experiment.py)**: 实验入口，负责环境、代理、训练器的全生命周期管理。
- **[world.py](file:///d:/Axon/ANN/tianshou/tianshou/highlevel/world.py)**: 运行时上下文容器，存储环境、策略、采集器等实例。
- **params/**: 算法超参数的 Dataclass 定义及自动转换工具（ParamTransformer）。
- **module/**: 网络结构的工厂类（Actor/Critic Factory）。

### 3. [data/](file:///d:/Axon/ANN/tianshou/tianshou/data) —— 数据的“血脉” (The Data Flow)
负责经验的存储、采样与流转。

- **[batch.py](file:///d:/Axon/ANN/tianshou/tianshou/data/batch.py)**: 核心数据结构 `Batch`，支持递归操作。
- **buffer/**: 各种重放缓存实现（Vector, Prioritized, HER 等）。
- **collector.py**: 采集器，连接环境与缓存的桥梁。

### 4. [env/](file:///d:/Axon/ANN/tianshou/tianshou/env) —— 交互的“世界” (The Environment)
对 Gymnasium 环境的封装与并行化支持。

- **venvs.py**: 矢量化环境环境实现。
- **worker/**: 定义了并行环境的具体工作模式（Subprocess, Ray, Dummy）。
- **wrappers/**: 常见的环境包装器（Atari, PettingZoo 等）。

### 5. [utils/](file:///d:/Axon/ANN/tianshou/tianshou/utils) —— 通用的“工具” (The Toolbox)
底层支撑模块。

- **net/**: 通用的网络组件（MLP, CNN, RNN）。
- **logger/**: 多后端日志支持（TensorBoard, WandB）。
- **torch_utils.py**: PyTorch 相关辅助函数。

---

## 💡 核心设计哲学

1. **组合优于继承 (Composition over Inheritance)**:
   `Algorithm` 类通过组合一个 `Policy` 对象来工作，而不是像 1.0 那样让所有逻辑挤在一个类里。
2. **声明式 vs 命令式**:
   通过 `highlevel` 模块，用户只需声明“想要什么”（配置），而无需编写“如何去做”（初始化过程）。
3. **强类型约束**:
   大量使用 Python 类型提示 (Type Hinting) 和 `Protocol`，确保在编译期就能发现大部分架构层面的错误。

---

## 🚀 开发者复刻指南

若要复刻或扩展 Tianshou 2.0：
1. **新增网络**: 在 `utils/net` 中定义。
2. **新增策略**: 继承 `algorithm/algorithm_base.py` 中的 `Policy`。
3. **新增算法**: 继承 `Algorithm` 并实现其 `learn` 或 `update` 逻辑。
4. **集成高阶 API**: 在 `highlevel/params` 中定义参数结构，并实现对应的 `AgentFactory`。

---
*Created by AI Pair Programmer for Tianshou 2.0 Enthusiasts.*
