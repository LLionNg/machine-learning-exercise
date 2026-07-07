from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 1):
        super().__init__()
        # Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, output_dim)
        # TODO: define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return the raw logits (no sigmoid)
        # TODO
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)                        # raw logits


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
          epochs: int = 20, lr: float = 0.1) -> list[float]:
    # BCEWithLogitsLoss + Adam; loop `epochs` times; return the per-epoch losses
    # TODO: implement the training loop
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()      # clear gradients from the previous step
        logits = model(X)          # forward pass
        loss = loss_fn(logits, y)  # compare to targets
        loss.backward()            # backprop: fill each parameter's .grad
        optimizer.step()           # nudge parameters down the gradient
        history.append(loss.item())  # .item() -> Python float
    return history