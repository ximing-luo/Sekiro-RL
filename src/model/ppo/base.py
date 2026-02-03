import torch
import torch.nn as nn
from typing import List, Dict, Union, Type
from stable_baselines3.common.torch_layers import MlpExtractor
from ..components import RMSNorm

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
    正常 PPO MLP 提取器 (带 RMSNorm)
    """
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module] = nn.SiLU,
        device: Union[torch.device, str] = "auto",
    ):
        nn.Module.__init__(self)
        
        device = torch.device(device) if isinstance(device, str) and device != "auto" else device
        
        policy_arch, value_arch = [], []
        if isinstance(net_arch, list):
            policy_arch, value_arch = net_arch, net_arch
        else:
            policy_arch = net_arch.get("pi", [])
            value_arch = net_arch.get("vf", [])

        self.latent_dim_pi = policy_arch[-1] if len(policy_arch) > 0 else feature_dim
        self.latent_dim_vf = value_arch[-1] if len(value_arch) > 0 else feature_dim

        self.policy_net = self._build_network(feature_dim, policy_arch)
        self.value_net = self._build_network(feature_dim, value_arch)

    def _build_network(self, input_dim: int, arch: List[int]) -> nn.Sequential:
        layers = []
        curr_dim = input_dim
        for next_dim in arch:
            layers.append(NormalizedMLP(curr_dim, next_dim))
            curr_dim = next_dim
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features), self.value_net(features)
