import torch

from solution import (
    add_vector_to_rows,
    get_column,
    matmul,
    reshape,
    row_sums,
    tensor_info,
    to_tensor,
)


def test_to_tensor_type_and_values():
    t = to_tensor([[1, 2], [3, 4]])
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert t.shape == (2, 2)
    assert torch.allclose(t, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_to_tensor_is_float_even_from_ints():
    t = to_tensor([1, 2, 3])
    assert t.dtype == torch.float32
    assert list(t.shape) == [3]


def test_tensor_info():
    info = tensor_info(torch.zeros(2, 3))
    assert info["shape"] == (2, 3)
    assert info["ndim"] == 2
    assert info["numel"] == 6
    assert info["dtype"] == "torch.float32"


def test_tensor_info_1d():
    info = tensor_info(torch.arange(5))
    assert info["shape"] == (5,)
    assert info["ndim"] == 1
    assert info["numel"] == 5


def test_reshape_shape_and_values():
    r = reshape(torch.arange(6, dtype=torch.float32), 2, 3)
    assert r.shape == (2, 3)
    assert torch.allclose(r, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))


def test_reshape_preserves_row_major_order():
    r = reshape(torch.arange(12, dtype=torch.float32), 3, 4)
    assert r.shape == (3, 4)
    assert r[1, 0].item() == 4.0  # row-major: element 4 starts the second row


def test_get_column():
    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    col = get_column(t, 1)
    assert col.shape == (2,)
    assert torch.allclose(col, torch.tensor([2.0, 5.0]))


def test_row_sums():
    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    s = row_sums(t)
    assert s.shape == (2,)
    assert torch.allclose(s, torch.tensor([6.0, 15.0]))


def test_add_vector_to_rows_broadcasts():
    m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = torch.tensor([10.0, 20.0, 30.0])
    out = add_vector_to_rows(m, v)
    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.tensor([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]))


def test_matmul_square():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    out = matmul(a, b)
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[19.0, 22.0], [43.0, 50.0]]))


def test_matmul_nonsquare_shapes():
    out = matmul(torch.ones(2, 3), torch.ones(3, 4))
    assert out.shape == (2, 4)
    assert torch.allclose(out, torch.full((2, 4), 3.0))
