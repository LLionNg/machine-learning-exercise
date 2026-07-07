from __future__ import annotations

import torch


def init_params(seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    w1 = torch.randn(1, 32, requires_grad=True)
    b1 = torch.randn(32, requires_grad=True)
    w2 = torch.randn(32, 16, requires_grad=True)
    b2 = torch.randn(16, requires_grad=True)
    w3 = torch.randn(16, 1, requires_grad=True)
    b3 = torch.randn(1, requires_grad=True)
    return [w1, b1, w2, b2, w3, b3]


def forward(params: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2, w3, b3 = params
    x = torch.sigmoid(x @ w1 + b1)
    x = torch.sigmoid(x @ w2 + b2)
    return x @ w3 + b3


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    e = pred - target
    return torch.mean(e * e)


def train(
    params: list[torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.1,
) -> list[float]:
    history: list[float] = []
    for _ in range(epochs):
        loss = mse(forward(params, x), y)
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= p.grad * lr
                p.grad.zero_()
        history.append(loss.item())
    return history
