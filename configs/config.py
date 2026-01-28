"""
项目统一配置

模块用途：集中管理运行时的基础参数，供环境、训练与可视化模块统一引用。

包含：常量配置项

边界：
- 负责：参数值与默认值的声明
- 不负责：业务逻辑、状态存储或动态变更
"""

IMG_WIDTH = 480  # 代理状态输入图像宽度（像素）
IMG_HEIGHT = 270  # 代理状态输入图像高度（像素）
FRAME_HISTORY_LEN = 12  # 回放缓冲帧堆叠长度（每个状态包含的连续帧数）
CAPTURE_FPS = 60  # 摄像头采集帧率（FPS）
CAMERA_INDEX = 1  # 摄像头设备索引（Windows 下常用 0/1）
CAMERA_WIDTH = 1920  # 摄像头采集分辨率宽度
CAMERA_HEIGHT = 1080  # 摄像头采集分辨率高度
BLOOD_WINDOW = (110, 90, 625, 907)  # 血量区域裁剪窗口 (x1,y1,x2,y2)
STAMINA_WINDOW = (586, 54, 750, 900)  # 架势条区域裁剪窗口 (x1,y1,x2,y2)
DEBUG_VIS_FPS = 60  # 环境输入调试窗口显示帧率
MODEL_PATH = "models/dqn_model.pth"  # 模型权重保存路径
LOG_DIR = "logs"  # 训练日志保存路径

# 训练超参数
LR = 0.0001  # 学习率（Adam）
GRAD_CLIP_NORM = 10.0  # 梯度裁剪阈值（L2范数上限），建议 5.0~10.0
TARGET_UPDATE_FREQ = 25  # 目标网络同步频率（以优化步数计）
SAVE_FREQ = 100  # 模型保存频率（以优化步数计）
GAMMA = 0.99
MICRO_BATCH_SIZE = 2  # 单次采样的最小批大小（用于采样与AMP）
BATCH_SIZE = 8  # 逻辑训练批大小（影响梯度累积）

OPTIMIZE_EVERY_STEPS = 5  # 每 N 步优化一次模型（建议 1~5）

# 优先经验回放（PER）参数
PER_ALPHA = 0.6  # 采样分布偏向强度，0=均匀，1=完全按优先度
PER_BETA_START = 0.4  # 重要性采样权重初始值，抵消偏置
PER_BETA_END = 1.0    # 训练后期权重目标值
PER_BETA_STEPS = 200000  # 线性提升到 PER_BETA_END 的步数（按优化步）