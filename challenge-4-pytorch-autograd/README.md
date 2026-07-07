# Challenge 4: A Neural Network from Scratch with PyTorch Autograd

**Difficulty:** Beginner-Intermediate | **Est. time:** 40-50 min | **Topic:** `requires_grad`, `loss.backward()`, manual gradient descent

Challenge 3 built a network with `nn.Module` and let `torch.optim.Adam` do the
learning. This challenge pulls back the curtain: you build the **same kind of
network by hand** - raw weight tensors, your own matmuls, and a gradient-descent
loop you write yourself. No `nn.Module`, no `nn.Linear`, no optimizer.

This is the single most useful thing to understand about PyTorch: a `Tensor` with
`requires_grad=True` records the operations you do to it, and `loss.backward()`
walks that record backwards to fill in every tensor's `.grad`. Everything else
(`nn.Linear`, `Adam`) is convenience built on top.

The problem is a small **regression**: fit `y = sin(3x + 1) + noise` with an MLP
of shape `[1, 32, 16, 1]`. As in Challenge 3, the tests **don't** train to
convergence - they check the pieces are wired up and that a few hundred manual
steps move the loss **down**. Fast and CPU-only.

## What to implement

### `init_params(seed=0) -> list[torch.Tensor]`

Return the six **leaf** tensors `[w1, b1, w2, b2, w3, b3]`, each created with
`requires_grad=True`. Seed torch first (`torch.manual_seed(seed)`) so runs are
reproducible.

| Tensor | Shape |
|---|---|
| `w1` | `(1, 32)` |
| `b1` | `(32,)` |
| `w2` | `(32, 16)` |
| `b2` | `(16,)` |
| `w3` | `(16, 1)` |
| `b3` | `(1,)` |

### `forward(params, x) -> torch.Tensor`

One forward pass through three layers. Input `x` is `(N, 1)`; each row below is one
step:

| Step | Operation | Output shape |
|---|---|---|
| 1 | `sigmoid(x @ w1 + b1)` | `(N, 32)` |
| 2 | `sigmoid(x @ w2 + b2)` | `(N, 16)` |
| 3 | `x @ w3 + b3` (no activation) | `(N, 1)` |

The first two layers apply `sigmoid` to add non-linearity. The output layer is
**linear** - it has no activation, because regression targets can be negative and
the output must be free to be any real number.

### `mse(pred, target) -> torch.Tensor`

Mean squared error: `mean((pred - target) ** 2)`. Return the scalar tensor (so
`.backward()` can be called on it).

### `train(params, x, y, epochs=1000, lr=0.1) -> list[float]`

The manual training loop. Each epoch, in order:

1. `loss = mse(forward(params, x), y)`
2. `loss.backward()` - fills each parameter's `.grad`
3. `with torch.no_grad():` update **every** parameter `p -= p.grad * lr`
4. still inside `no_grad()`, `p.grad.zero_()` - reset before the next step

Append `loss.item()` (a Python float) to a history list and return it
(`len == epochs`).

## How the tests check your work

1. **The raw tensors** - `init_params` returns six leaf tensors of the right
   shapes, each with `requires_grad=True`.
2. **Data flows through (forward)** - `(N, 1)` in gives finite `(N, 1)` out.
3. **`mse` is correct** - a known input gives the exact expected value.
4. **Gradients flow back** - after one `backward()`, the parameters have
   non-`None`, non-zero gradients.
5. **The loop learns** - over ~300 manual steps the loss goes **down** (never
   `NaN`), the weights actually change, and the gradients are zeroed each step.

## Run the tests

```bash
uv sync                                # from the repo root, once
mkdir -p submissions/<your-username>
cp solution_template.py submissions/<your-username>/solution.py
# ...implement it, then:
./run_tests.sh                         # enter your username when prompted
```

See `learning.md` for the autograd primer and `hints.md` if you get stuck.
