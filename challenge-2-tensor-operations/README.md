# Challenge 2: PyTorch Tensor Operations

**Difficulty:** Beginner | **Est. time:** 30-45 min | **Topic:** PyTorch tensors

Before you can build a neural network, you need to be comfortable moving data
around with **tensors**. A tensor is PyTorch's core data structure: an
n-dimensional array (much like a NumPy array) that also runs on GPUs and can
track gradients for training.

This challenge is a set of small, self-contained functions - each one practises
one fundamental tensor operation. Everything is tiny and CPU-only.

## What is a tensor? (quick version)

| Dimensions | Name | Example shape |
|---|---|---|
| 0-D | scalar | `()` |
| 1-D | vector | `(3,)` |
| 2-D | matrix | `(2, 3)` |
| n-D | tensor | `(8, 3, 32, 32)` |

Every tensor has a **shape** (size along each dimension) and a **dtype** (e.g.
`torch.float32`). See `learning.md` for a full primer.

## What to implement

| Function | What it does |
|---|---|
| `to_tensor(data)` | Build a **float32** tensor from a (possibly nested) list |
| `tensor_info(t)` | Return `{"shape": tuple, "ndim": int, "numel": int, "dtype": str}` |
| `reshape(t, rows, cols)` | Reshape `t` to `(rows, cols)` |
| `get_column(t, j)` | Return column `j` as a 1-D tensor |
| `row_sums(t)` | Sum each row -> 1-D tensor (reduce along columns) |
| `add_vector_to_rows(matrix, vector)` | Add a length-`n` vector to every row via **broadcasting** |
| `matmul(a, b)` | **Matrix** product of `a` and `b` (not element-wise) |

## How the tests check your work

Each function is checked for the correct **shape** and **values** on small
fixed inputs (`torch.allclose` for values). There's no randomness and no
training - a correct implementation passes instantly.

Common gotchas the tests catch:
- `to_tensor` must produce **float32** (not int64) even when the input is ints.
- `matmul` must be matrix multiplication (`@` / `torch.matmul`), **not** `*`
  (which is element-wise).
- `row_sums` reduces along **columns** (`dim=1`), giving one number per row.

## Run the tests

```bash
# from the repo root, once: install everything with uv
uv sync

mkdir -p submissions/<your-username>
cp solution_template.py submissions/<your-username>/solution.py
# ...implement it, then:
./run_tests.sh          # enter your username when prompted
```

See `learning.md` for the tensor primer and `hints.md` if you get stuck.
