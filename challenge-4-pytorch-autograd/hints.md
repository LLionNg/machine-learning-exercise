# Hints - Challenge 4

Reveal these one at a time - try each step before the next hint.

## Hint 1: Seed, then build leaf tensors
The first line of `init_params` should be `torch.manual_seed(seed)`. Then create
each tensor with `torch.randn(<shape>, requires_grad=True)`. Because you pass
`requires_grad=True` to the constructor, the result is a **leaf** the optimizer
can learn.

## Hint 2: Match the shapes exactly
`w1 (1, 32)`, `b1 (32,)`, `w2 (32, 16)`, `b2 (16,)`, `w3 (16, 1)`, `b3 (1,)`.
Return them as a list in that order: `[w1, b1, w2, b2, w3, b3]`.

## Hint 3: The forward pass is three lines
```python
x = torch.sigmoid(x @ w1 + b1)
x = torch.sigmoid(x @ w2 + b2)
return x @ w3 + b3
```
`@` is matrix multiply; the bias broadcasts across the batch. The last line has
**no** sigmoid - the output must be able to be negative.

## Hint 4: `mse` returns a scalar
`return torch.mean((pred - target) ** 2)`. It must be a single 0-dim tensor so you
can call `.backward()` on it.

## Hint 5: The loop has four steps
Forward -> backward -> update -> zero, every epoch:
```python
loss = mse(forward(params, x), y)
loss.backward()
with torch.no_grad():
    for p in params:
        p -= p.grad * lr
        p.grad.zero_()
history.append(loss.item())
```

## Hint 6: Why `torch.no_grad()`?
The update `p -= p.grad * lr` is itself math on grad-tracking tensors. Wrap it in
`with torch.no_grad():` so PyTorch doesn't record the update into the graph. Do it
in place (`-=`), not `p = p - p.grad * lr` (which would make a new non-leaf
tensor and break the next `backward()`).

## Hint 7: Don't forget `grad.zero_()`
PyTorch **adds** into `.grad` each `backward()`. If you never reset it, gradients
pile up across epochs and the loss won't go down cleanly. Zero every parameter's
grad each step (inside the `no_grad()` block).

## Hint 8: Return Python floats
The tests expect `train(...)` to return `list[float]`. Append `loss.item()` (which
converts a 0-dim tensor to a Python float), not `loss` itself.

## Hint 9: Gradient-flow test failing?
Make sure every parameter is actually used in `forward` (all six appear in the
matmuls), and that you didn't wrap the forward pass in `no_grad()`. Only the
**update** goes inside `no_grad()`.

## Hint 10: Loss not going down (or NaN)?
Check that you zero the grads each step, that the update uses `-=` (downhill, not
uphill), and that the output layer is linear. If the loss explodes to `NaN`, your
learning rate/update sign is likely wrong or grads are accumulating.
