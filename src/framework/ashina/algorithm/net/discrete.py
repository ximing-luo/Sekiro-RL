from typing import Sequence
import torch
import torch.nn as nn
from .common import MLP


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
