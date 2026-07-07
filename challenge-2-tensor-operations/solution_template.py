from __future__ import annotations

import torch


def to_tensor(data: list) -> torch.Tensor:
    # build a float32 tensor from a (possibly nested) Python list
    # TODO
    raise NotImplementedError


def tensor_info(t: torch.Tensor) -> dict:
    # return {"shape": tuple, "ndim": int, "numel": int, "dtype": str}
    # TODO
    raise NotImplementedError


def reshape(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    # reshape `t` to (rows, cols)
    # TODO
    raise NotImplementedError


def get_column(t: torch.Tensor, j: int) -> torch.Tensor:
    # return column `j` of `t` as a 1-D tensor
    # TODO
    raise NotImplementedError


def row_sums(t: torch.Tensor) -> torch.Tensor:
    # return each row's sum (reduce along columns) as a 1-D tensor
    # TODO
    raise NotImplementedError


def add_vector_to_rows(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    # add `vector` (n,) to every row of `matrix` (m, n) via broadcasting
    # TODO
    raise NotImplementedError


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # matrix product `a @ b` (not element-wise)
    # TODO
    raise NotImplementedError
