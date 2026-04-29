from typing import Sequence
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(activation())
        layers.pop()  # 移除最后一层的激活函数
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
