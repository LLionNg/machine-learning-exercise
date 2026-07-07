"""Reference solution for Challenge 2: PyTorch tensor operations."""

from __future__ import annotations

import torch


def to_tensor(data: list) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32)


def tensor_info(t: torch.Tensor) -> dict:
    return {
        "shape": tuple(t.shape),
        "ndim": t.ndim,
        "numel": t.numel(),
        "dtype": str(t.dtype),
    }


def reshape(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return t.reshape(rows, cols)


def get_column(t: torch.Tensor, j: int) -> torch.Tensor:
    return t[:, j]


def row_sums(t: torch.Tensor) -> torch.Tensor:
    return t.sum(dim=1)


def add_vector_to_rows(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return matrix + vector


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b
