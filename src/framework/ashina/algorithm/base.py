from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from ..data.batch import Batch

class Policy(nn.Module, ABC):
    """
    策略基类 (Data Plane)
    负责推理过程，将观测转换为动作分布或具体动作。
    """
    def __init__(self, action_dim: int, device: Union[str, torch.device] = "cuda"):
        super().__init__()
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.training = True  # 记录当前模式

    def train(self, mode: bool = True):
        """设置训练/评估模式。"""
        self.training = mode
        super().train(mode)

    def eval(self):
        """设置评估模式。"""
        self.train(False)

    @abstractmethod
    def forward(
        self, 
        batch: Batch, 
        state: Optional[Any] = None, 
        **kwargs: Any
    ) -> Batch:
        """
        前向推理过程。
        :param batch: 包含 obs 的数据批。
        :param state: 隐藏状态（用于 RNN 等）。
        :return: 包含 logits/act 等信息的 Batch。
        """
        pass

class Algorithm(ABC):
    """
    算法基类 (Control Plane)
    负责学习逻辑，管理策略、优化器和超参数。
    """
    def __init__(self, policy: Policy, device: Union[str, torch.device] = "cuda"):
        self.policy = policy
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    @abstractmethod
    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        """
        核心学习逻辑。
        :param batch: 从 Buffer 采样的数据。
        :return: 包含 loss、td_error 等训练统计信息的字典。
        """
        pass

    def update(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        """
        更新接口，通常调用 learn。
        """
        return self.learn(batch, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """代理调用 policy 的 forward"""
        return self.policy(*args, **kwargs)
