from __future__ import annotations

import torch


def init_params(seed: int = 0) -> list[torch.Tensor]:
    # seed torch, then create the six leaf tensors [w1, b1, w2, b2, w3, b3] for a
    # [1, 32, 16, 1] MLP, each with requires_grad=True (see README for shapes)
    # TODO
    torch.manual_seed(seed)
    w1 = torch.randn(1, 32, requires_grad=True)
    b1 = torch.randn(32, requires_grad=True)
    w2 = torch.randn(32, 16, requires_grad=True)
    b2 = torch.randn(16, requires_grad=True)
    w3 = torch.randn(16, 1, requires_grad=True)
    b3 = torch.randn(1, requires_grad=True)
    return [w1, b1, w2, b2, w3, b3]


def forward(params: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    # two hidden layers with sigmoid, then a linear output; maps (N, 1) to (N, 1)
    # TODO
    w1, b1, w2, b2, w3, b3 = params
    x = torch.sigmoid(x @ w1 + b1)   # (N,1) @ (1,32) -> (N,32), + bias, squashed
    x = torch.sigmoid(x @ w2 + b2)   # (N,32) @ (32,16) -> (N,16)
    return x @ w3 + b3               # (N,16) @ (16,1) -> (N,1), LINEAR output



def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # mean squared error: the mean of (pred - target) squared
    # TODO
    e = pred - target
    return torch.mean(e * e)         # a 0-dim tensor


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
    history = []
    for _ in range(epochs):
        loss = mse(forward(params, x), y)   # forward
        loss.backward()                     # backward: fill .grad
        with torch.no_grad():
            for p in params:
                p -= p.grad * lr            # update
                p.grad.zero_()              # reset
        history.append(loss.item())         # .item() -> Python float
    return history
