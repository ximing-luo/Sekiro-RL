import torch
import torch.nn as nn
from typing import List, Dict, Union, Any, Type
from stable_baselines3.common.torch_layers import MlpExtractor
from .backbone.rms import RMSNorm

class NormalizedMLP(nn.Module):
    """
    带归一化的 MLP 模块 (LLM 风格)。
    结构: Linear -> RMSNorm -> SiLU
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = RMSNorm(output_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.linear(x)))

class SekiroMLPExtractor(MlpExtractor):
    """
    自定义 MLP 提取器，为 Policy 和 Value 分支的每一层添加 RMSNorm。
    """
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module] = nn.SiLU, # 默认使用 SiLU
        device: Union[torch.device, str] = "auto",
    ):
        # 我们不调用 super().__init__，因为我们要完全重写构建逻辑
        nn.Module.__init__(self)
        
        device = torch.device(device) if isinstance(device, str) and device != "auto" else device
        
        policy_arch, value_arch = [], []
        if isinstance(net_arch, list):
            policy_arch, value_arch = net_arch, net_arch
        else:
            policy_arch = net_arch.get("pi", [])
            value_arch = net_arch.get("vf", [])

        # 保存维度供 Policy 使用
        self.latent_dim_pi = policy_arch[-1] if len(policy_arch) > 0 else feature_dim
        self.latent_dim_vf = value_arch[-1] if len(value_arch) > 0 else feature_dim

        # 构建 Policy 网络
        self.policy_net = self._build_network(feature_dim, policy_arch)
        # 构建 Value 网络
        self.value_net = self._build_network(feature_dim, value_arch)

    def _build_network(self, input_dim: int, arch: List[int]) -> nn.Sequential:
        layers = []
        curr_dim = input_dim
        for next_dim in arch:
            layers.append(NormalizedMLP(curr_dim, next_dim))
            curr_dim = next_dim
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        返回 (latent_policy, latent_value)
        """
        return self.policy_net(features), self.value_net(features)
