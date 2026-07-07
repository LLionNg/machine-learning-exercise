from __future__ import annotations

import torch


def init_params(seed: int = 0) -> list[torch.Tensor]:
    # seed torch, then create the six leaf tensors [w1, b1, w2, b2, w3, b3] for a
    # [1, 32, 16, 1] MLP, each with requires_grad=True (see README for shapes)
    # TODO
    raise NotImplementedError


def forward(params: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    # two hidden layers with sigmoid, then a linear output; maps (N, 1) to (N, 1)
    # TODO
    raise NotImplementedError


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # mean squared error: the mean of (pred - target) squared
    # TODO
    raise NotImplementedError


def train(
    params: list[torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.1,
) -> list[float]:
    # manual gradient descent; each epoch: forward, backward, update every param
    # under no_grad, then zero its grad. Return the per-epoch loss as floats
    # TODO
    raise NotImplementedError
