# Challenge 5: A Neural Network from Scratch with TensorFlow GradientTape

**Difficulty:** Beginner-Intermediate | **Est. time:** 40-50 min | **Topic:** `tf.Variable`, `tf.GradientTape`, `assign_sub`

This is the **TensorFlow counterpart of Challenge 4**. Same network, same
problem, same from-scratch spirit - but with TensorFlow's autodiff instead of
PyTorch's. No Keras layers, no Keras optimizer, no `model.fit`.

The core idea: a `tf.GradientTape` **records** the math you do inside its `with`
block, and afterwards `tape.gradient(loss, variables)` differentiates the loss
with respect to your variables. You then apply the update yourself with
`Variable.assign_sub`.

The problem is the same small **regression**: fit `y = sin(3x + 1) + noise` with
an MLP of shape `[1, 32, 16, 1]`. The tests **don't** train to convergence - they
check the pieces are wired up and that a few hundred manual steps move the loss
**down**. Fast and CPU-only.

> Doing Challenge 4 first is recommended - comparing the two makes both
> frameworks click. The big difference: PyTorch accumulates gradients on the
> tensors (so you call `grad.zero_()`), while a `GradientTape` is fresh each
> block, so there's nothing to zero.

## What to implement

### `init_params(seed=0) -> list[tf.Variable]`

Return the six trainable variables `[w1, b1, w2, b2, w3, b3]`, each a
`tf.Variable` wrapping `tf.random.normal(<shape>)`. Seed TF first
(`tf.random.set_seed(seed)`).

| Variable | Shape |
|---|---|
| `w1` | `(1, 32)` |
| `b1` | `(32,)` |
| `w2` | `(32, 16)` |
| `b2` | `(16,)` |
| `w3` | `(16, 1)` |
| `b3` | `(1,)` |

### `forward(params, x) -> tf.Tensor`

One forward pass through three layers. Input `x` is `(N, 1)`; each row below is one
step:

| Step | Operation | Output shape |
|---|---|---|
| 1 | `sigmoid(x @ w1 + b1)` | `(N, 32)` |
| 2 | `sigmoid(x @ w2 + b2)` | `(N, 16)` |
| 3 | `x @ w3 + b3` (no activation) | `(N, 1)` |

The first two layers apply `sigmoid` to add non-linearity. The output layer is
**linear** - it has no activation, because regression targets can be negative.

### `mse(pred, target) -> tf.Tensor`

Mean squared error: `tf.reduce_mean((pred - target) ** 2)`.

### `train(params, x, y, epochs=1000, lr=0.1) -> list[float]`

The manual training loop. Each epoch:

1. inside `with tf.GradientTape() as tape:` compute `loss = mse(forward(params, x), y)`
2. `grads = tape.gradient(loss, params)`
3. `for v, g in zip(params, grads): v.assign_sub(g * lr)`

Append `float(loss.numpy())` to a history list and return it (`len == epochs`).

## How the tests check your work

1. **The variables** - `init_params` returns six trainable `tf.Variable`s of the
   right shapes.
2. **Data flows through (forward)** - `(N, 1)` in gives finite `(N, 1)` out.
3. **`mse` is correct** - a known input gives the exact expected value.
4. **Gradients flow** - `tape.gradient` returns a non-`None`, non-zero gradient
   for every variable.
5. **The loop learns** - over ~300 manual steps the loss goes **down** (never
   `NaN`) and the weights actually change.

## Run the tests

```bash
uv sync                                # from the repo root, once: installs tensorflow + pytest
mkdir -p submissions/<your-username>
cp solution_template.py submissions/<your-username>/solution.py
# ...implement it, then:
./run_tests.sh                         # enter your username when prompted
```

See `learning.md` for the GradientTape primer and `hints.md` if you get stuck.
