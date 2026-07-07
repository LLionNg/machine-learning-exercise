from __future__ import annotations

import torch
from torch import nn


def build_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.Sigmoid(),
        nn.Linear(32, 16),
        nn.Sigmoid(),
        nn.Linear(16, 1),
    )


def predict(model: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)
