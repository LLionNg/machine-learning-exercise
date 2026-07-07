# Learning: PyTorch autograd from the ground up

This follows PyTorch's own
["A Gentle Introduction to `torch.autograd`"](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
and the manual-gradient example in
["Learning PyTorch with Examples"](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html),
trimmed to exactly what this challenge needs.

## 1. `requires_grad`: a tensor that remembers

A normal tensor is just numbers. Set `requires_grad=True` and PyTorch starts
**recording** every operation you perform on it, building a graph it can later
differentiate:

```python
import torch
w = torch.randn(1, 32, requires_grad=True)   # a leaf tensor we want to learn
```

- A **leaf** tensor is one you created directly (not the result of an op on other
  grad-tracking tensors). Your weights and biases are leaves - they're the knobs
  gradient descent turns. `w.is_leaf` is `True`.
- Ops on it produce non-leaf tensors that carry a `grad_fn` (the recorded
  operation), e.g. `(x @ w).grad_fn`.

## 2. The forward pass is just tensor math

There's no magic layer object. `nn.Linear(in, out)` is literally `x @ W + b` with
a stored `W` and `b`. Here you write it yourself:

```python
def forward(params, x):
    w1, b1, w2, b2, w3, b3 = params
    x = torch.sigmoid(x @ w1 + b1)   # (N,1) @ (1,32) -> (N,32), + bias, squashed
    x = torch.sigmoid(x @ w2 + b2)   # (N,32) @ (32,16) -> (N,16)
    return x @ w3 + b3               # (N,16) @ (16,1) -> (N,1), LINEAR output
```

- `@` is matrix multiplication. The bias broadcasts across the batch.
- **`sigmoid`** squashes to `(0, 1)` and provides the non-linearity between
  layers. Two linear layers with nothing between them collapse into one linear
  layer, so the non-linearity is what makes it a real network.
- The **last layer is linear**. `y = sin(...)` takes negative and positive
  values, and a sigmoid output could only ever produce `(0, 1)` - so the final
  layer must be free to output any real number.

## 3. The loss must be a scalar

`backward()` is defined for a single number. Reduce your errors to one scalar:

```python
def mse(pred, target):
    e = pred - target
    return torch.mean(e * e)         # a 0-dim tensor
```

## 4. `backward()`: fill in the gradients

Call `.backward()` on the loss. PyTorch walks the recorded graph from the loss
back to every leaf and **accumulates** the gradient into each leaf's `.grad`:

```python
loss = mse(forward(params, x), y)
loss.backward()
w1.grad          # dLoss/dw1, same shape as w1
```

Before `backward()`, `.grad` is `None`. After, each parameter that contributed to
the loss has a gradient telling you which direction increases the loss.

## 5. The update: step *against* the gradient, with autograd off

Gradient descent nudges each parameter a little bit **downhill** - the opposite of
its gradient, scaled by a learning rate `lr`:

```python
with torch.no_grad():            # don't record THIS math into the graph
    for p in params:
        p -= p.grad * lr         # move downhill
        p.grad.zero_()           # reset for the next step
```

Two things people get wrong here:

- **`torch.no_grad()`** - the weight update is itself tensor math on
  grad-tracking tensors. If you don't disable autograd, PyTorch would record the
  update into the graph and you'd be differentiating your optimizer. `no_grad()`
  says "just change the numbers."
- **`grad.zero_()`** - PyTorch *accumulates* gradients (it adds into `.grad`
  rather than replacing it). If you never zero them, epoch 2's gradient is
  `grad1 + grad2`, epoch 3's is `grad1 + grad2 + grad3`, and training goes
  haywire. `optimizer.zero_grad()` from Challenge 3 did exactly this for you.

## 6. Putting it together: the training loop

```python
history = []
for _ in range(epochs):
    loss = mse(forward(params, x), y)   # forward
    loss.backward()                     # backward: fill .grad
    with torch.no_grad():
        for p in params:
            p -= p.grad * lr            # update
            p.grad.zero_()              # reset
    history.append(loss.item())         # .item() -> Python float
```

That's the entire skeleton every deep-learning framework wraps. `nn.Module` gives
you the parameters and forward; `torch.optim` gives you steps 3-4. Once you've
written it by hand, those abstractions stop being mysterious.

## 7. How we verify without training to convergence

Same philosophy as Challenge 3 - cheap, standard sanity checks:

- **Forward shape & finiteness**: `(N,1)` in, finite `(N,1)` out.
- **Gradients flow**: after one `backward()`, every parameter has a non-`None`,
  non-zero `.grad`. If a parameter is missing from the forward pass, or you
  detached the graph, its gradient never arrives.
- **A few steps reduce the loss**: run a few hundred manual steps and confirm the
  loss moves down. You don't need a perfect fit - just proof the loop works.

## References

- A Gentle Introduction to `torch.autograd`: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- Learning PyTorch with Examples (manual autograd): https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
- Autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html
