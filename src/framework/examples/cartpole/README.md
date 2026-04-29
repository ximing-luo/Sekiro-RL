# CartPole DQN 示例（迭代式并行训练）

基于 `CartPole-v1` 环境演示**迭代式并行** DQN 训练与推理。

## 环境要求

```bash
conda activate lab
```

## 训练

**高层接口（默认，推荐）：**

```bash
python src/framework/examples/cartpole/train_cartpole.py --mode high
```

**低层手动组装接口：**

```bash
python src/framework/examples/cartpole/train_cartpole.py --mode low
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `high` | `high`=高层接口，`low`=手动组装 |
| `--num-iterations` | `100` | 总迭代次数 |
| `--steps-per-iter` | `24` | 每迭代每个环境跑的步数 |
| `--learn-epochs` | `4` | 每迭代学习轮次 |
| `--num-envs` | `1024` | 并行环境数（N 个 env 同时采集） |
| `--warmup` | `5000` | 预热步数（开始迭代前） |
| `--buffer-size` | `100000` | 经验回放缓冲区大小 |
| `--batch-size` | `2048` | 每次梯度更新的样本量 |
| `--lr` | `1e-3` | 学习率 |
| `--gamma` | `0.99` | 折扣因子 |
| `--target-update-freq` | `500` | 目标网络更新频率（步） |
| `--log-dir` | `logs/cartpole` | 模型保存根目录 |

```bash
# 1024 环境 × 24 步，batch_size=2048，每迭代 12 batch × 4 轮 = 48 次更新
python src/framework/examples/cartpole/train_cartpole.py --num-envs 1024 --steps-per-iter 24 --batch-size 2048 --num-iterations 1000
```

训练完成后模型自动保存至 `logs/cartpole/cartpole_dqn_<时间戳>/model.pth`。

## 推理（可视化）

```bash
python src/framework/examples/cartpole/play_cartpole.py --model logs/cartpole\cartpole_dqn_20260430_005706\model.pth
```

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `None` | 模型权重路径（`.pth`），不填则随机权重 |
| `--episodes` | `5` | 运行局数 |

> 所有命令均需在项目根目录 `d:\Axon\github\Sekiro-RL` 下执行。
