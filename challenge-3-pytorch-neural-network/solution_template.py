from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 1):
        super().__init__()
        # Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, output_dim)
        # TODO: define the layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return the raw logits (no sigmoid)
        # TODO
        raise NotImplementedError


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
          epochs: int = 20, lr: float = 0.1) -> list[float]:
    # BCEWithLogitsLoss + Adam; loop `epochs` times; return the per-epoch losses
    # TODO: implement the training loop
    raise NotImplementedError
