from __future__ import annotations

import torch
from torch import nn


def build_model() -> nn.Sequential:
    # a [1, 32, 16, 1] regression MLP as an nn.Sequential: Linear, Sigmoid,
    # Linear, Sigmoid, Linear (sigmoid activations, linear output)
    # TODO
    raise NotImplementedError


def predict(model: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    # run inference: put the model in eval mode and return its output without
    # tracking gradients; shape (N, 1)
    # TODO
    raise NotImplementedError
