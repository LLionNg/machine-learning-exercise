# Learning: TensorFlow autodiff from the ground up

This follows TensorFlow's own guides
["Introduction to gradients and automatic differentiation"](https://www.tensorflow.org/guide/autodiff)
and ["Basic training loops"](https://www.tensorflow.org/guide/basic_training_loops),
trimmed to exactly what this challenge needs. If you did Challenge 4, this is the
same story told with TensorFlow words.

## 1. `tf.Variable`: mutable, trainable state

A `tf.Tensor` is immutable. The knobs a network learns must be mutable, so
TensorFlow gives you `tf.Variable`:

```python
import tensorflow as tf
tf.random.set_seed(0)
w = tf.Variable(tf.random.normal((1, 32)))   # trainable by default
```

- `tf.random.normal(shape)` makes a constant tensor; wrapping it in
  `tf.Variable(...)` makes it a mutable, **trainable** parameter (`w.trainable`
  is `True`).
- Unlike PyTorch, you do **not** flag it for gradients at creation - the
  `GradientTape` decides what to watch (and it watches trainable variables
  automatically).

## 2. The forward pass is just tensor math

`keras.layers.Dense(units)` is `x @ W + b` under the hood. Here you write it:

```python
def forward(params, x):
    w1, b1, w2, b2, w3, b3 = params
    x = tf.sigmoid(x @ w1 + b1)   # (N,1) @ (1,32) -> (N,32), + bias, squashed
    x = tf.sigmoid(x @ w2 + b2)   # (N,32) @ (32,16) -> (N,16)
    return x @ w3 + b3            # (N,16) @ (16,1) -> (N,1), LINEAR output
```

- `@` is matrix multiply; the bias broadcasts across the batch.
- **`tf.sigmoid`** is the non-linearity between layers. Without it, stacked linear
  layers collapse into one.
- The **last layer is linear** - `y = sin(...)` can be negative, and a sigmoid
  output could only produce `(0, 1)`.

## 3. The loss is a scalar

```python
def mse(pred, target):
    e = pred - target
    return tf.reduce_mean(e * e)   # a scalar tensor
```

## 4. `tf.GradientTape`: record, then differentiate

This is the heart of it. Operations inside the `with` block are **recorded** onto
the tape; afterwards you ask the tape for gradients:

```python
with tf.GradientTape() as tape:
    loss = mse(forward(params, x), y)    # recorded
grads = tape.gradient(loss, params)      # dLoss/dparam for each param
```

- The tape automatically **watches trainable variables**, so you don't have to
  register them (contrast PyTorch's `requires_grad=True`).
- `tape.gradient(loss, params)` returns a list of gradients, one per variable,
  each the same shape as its variable.
- A tape is **single-use** by default: it's consumed by the first
  `tape.gradient(...)` call. That's fine - we make a fresh tape every epoch.

## 5. The update: `assign_sub`

Move each variable a little bit **downhill** - against its gradient, scaled by the
learning rate. Because a `Variable` is mutable, you update it in place:

```python
for v, g in zip(params, grads):
    v.assign_sub(g * lr)     # v <- v - g*lr
```

`assign_sub(x)` is `v -= x`. There's no `no_grad()` needed like in PyTorch: the
update happens **outside** the tape's `with` block, so it isn't recorded.

## 6. Two differences from PyTorch (Challenge 4)

If you did the PyTorch version, note:

| | PyTorch (ch. 4) | TensorFlow (ch. 5) |
|---|---|---|
| trainable tensor | `torch.randn(..., requires_grad=True)` | `tf.Variable(tf.random.normal(...))` |
| record the graph | always on for grad tensors | only inside `tf.GradientTape()` |
| get gradients | `loss.backward()` fills `.grad` | `tape.gradient(loss, vars)` returns them |
| update | `with torch.no_grad(): p -= p.grad*lr` | `v.assign_sub(g*lr)` |
| reset gradients | **`p.grad.zero_()` each step** | nothing - a fresh tape each block |

The last row is the classic gotcha: PyTorch *accumulates* into `.grad` so you must
zero it; a `GradientTape` starts empty every block, so there is nothing to reset.

## 7. Putting it together

```python
history = []
for _ in range(epochs):
    with tf.GradientTape() as tape:
        loss = mse(forward(params, x), y)   # forward, recorded
    grads = tape.gradient(loss, params)     # backward
    for v, g in zip(params, grads):
        v.assign_sub(g * lr)                # update
    history.append(float(loss.numpy()))     # .numpy() -> Python float
```

## 8. How we verify without training to convergence

Same cheap sanity checks as the PyTorch challenge:

- **Forward shape & finiteness**: `(N,1)` in, finite `(N,1)` out.
- **Gradients flow**: `tape.gradient` returns a non-`None`, non-zero gradient for
  every variable.
- **A few steps reduce the loss**: run a few hundred manual steps and confirm the
  loss moves down.

## References

- Introduction to gradients and automatic differentiation: https://www.tensorflow.org/guide/autodiff
- Basic training loops: https://www.tensorflow.org/guide/basic_training_loops
- `tf.GradientTape` API: https://www.tensorflow.org/api_docs/python/tf/GradientTape
