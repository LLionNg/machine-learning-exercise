# Learning: What is a tensor?

A **tensor** is PyTorch's fundamental data structure - an **n-dimensional
array**. If you've used NumPy arrays, tensors are the same idea, with two
superpowers: they can live on a **GPU**, and they can **track gradients** for
training. Everything in PyTorch - inputs, weights, activations, outputs - is a
tensor.

This primer follows the official
[Tensors tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).

## 1. Dimensions

The number of dimensions (the "rank") is what the *n* in n-dimensional means:

| Rank | Name | Example | Shape |
|---|---|---|---|
| 0 | scalar | `3.0` | `()` |
| 1 | vector | `[1, 2, 3]` | `(3,)` |
| 2 | matrix | `[[1, 2], [3, 4]]` | `(2, 2)` |
| 3+ | tensor | a batch of RGB images | `(8, 3, 32, 32)` |

## 2. Every tensor has a shape and a dtype

```python
import torch

t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
t.shape     # torch.Size([2, 3])  -> size along each dimension
t.ndim      # 2                   -> number of dimensions
t.numel()   # 6                   -> total number of elements
t.dtype     # torch.float32       -> element type
t.device    # device(type='cpu')  -> where it lives (cpu / cuda)
```

`torch.Size` behaves like a tuple, so `tuple(t.shape)` gives `(2, 3)`.

## 3. Creating tensors

```python
torch.tensor([[1, 2], [3, 4]])              # from a Python list
torch.tensor([1, 2], dtype=torch.float32)   # force a dtype
torch.zeros(2, 3)                           # all zeros, shape (2, 3)
torch.ones(2, 3)                            # all ones
torch.arange(6)                             # tensor([0, 1, 2, 3, 4, 5])
torch.rand(2, 3)                            # uniform random in [0, 1)
```

**dtype matters.** `torch.tensor([1, 2, 3])` infers `int64`; pass
`dtype=torch.float32` (or call `.float()`) when you need floats - most neural
network math is done in float32.

## 4. Indexing and slicing

Just like NumPy - `t[row, col]`, and `:` means "all of this dimension":

```python
t[0]        # first row              -> tensor([1., 2., 3.])
t[0, 1]     # element at row 0, col 1 -> tensor(2.)
t[:, 1]     # column 1 (all rows)     -> tensor([2., 5.])
t[1, :]     # row 1                   -> tensor([4., 5., 6.])
```

## 5. Reshaping

`reshape` (or `view`) rearranges the same data into a new shape. Data is laid
out **row-major**, so filling a `(2, 3)` from `[0,1,2,3,4,5]` gives rows
`[0,1,2]` and `[3,4,5]`:

```python
torch.arange(6).reshape(2, 3)
# tensor([[0, 1, 2],
#         [3, 4, 5]])
```

The total number of elements must stay the same (`numel` is unchanged).

## 6. Element-wise math

Arithmetic and most functions act **element by element**:

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])
a + b          # tensor([11., 22., 33.])
a * b          # tensor([10., 40., 90.])   <- element-wise product, NOT matmul
a ** 2         # tensor([1., 4., 9.])
torch.relu(a)  # applies max(0, x) to each element
```

## 7. Reductions

Reductions collapse a dimension. Use `dim=` to say **which** axis to reduce; the
result drops that dimension.

```python
t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
t.sum()          # tensor(21.)          -> everything
t.sum(dim=1)     # tensor([6., 15.])    -> sum across columns => one value per row
t.sum(dim=0)     # tensor([5., 7., 9.]) -> sum across rows    => one value per column
t.mean(dim=1)    # row means
```

Think of `dim=1` as "the columns dimension disappears," leaving one number per
row.

## 8. Broadcasting

Broadcasting lets tensors of different but compatible shapes combine without
manually copying data. Dimensions are matched from the right; a size-1 (or
missing) dimension is stretched to match:

```python
matrix = torch.ones(2, 3)              # shape (2, 3)
vector = torch.tensor([10., 20., 30.]) # shape (3,)
matrix + vector                        # vector is added to EACH row -> (2, 3)
# tensor([[11., 21., 31.],
#         [11., 21., 31.]])
```

This is exactly how a bias vector gets added to every row of a batch in a neural
network layer.

## 9. Matrix multiplication vs. element-wise

A frequent beginner bug: `*` is **element-wise**, while `@` (or
`torch.matmul`) is **matrix multiplication**.

```python
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])
a * b     # element-wise: [[5, 12], [21, 32]]
a @ b     # matmul:       [[19, 22], [43, 50]]
```

For matmul the inner dimensions must agree: `(m, k) @ (k, n) -> (m, n)`. This is
the operation at the heart of every `nn.Linear` layer, which you'll meet in the
next challenge.

## References

- Tensors: https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- `torch.Tensor` API: https://pytorch.org/docs/stable/tensors.html
- Broadcasting semantics: https://pytorch.org/docs/stable/notes/broadcasting.html
