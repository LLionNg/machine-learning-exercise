# Hints - Challenge 6

Reveal these one at a time - try each step before the next hint.

## Hint 1: Sequential takes the layers as arguments
Pass the modules to `nn.Sequential(...)` in order; it wires them front to back and
gives you `forward` for free:

```python
return nn.Sequential(
    nn.Linear(1, 32),
    nn.Sigmoid(),
    ...
)
```

## Hint 2: Alternate Linear and Sigmoid
The order is Linear, Sigmoid, Linear, Sigmoid, Linear. The two `Sigmoid`
activations sit **between** the Linear layers.

## Hint 3: The last layer is linear
There is **no** activation after the final `nn.Linear(16, 1)` - a regression
output must be able to be negative, so don't put a `Sigmoid` at the end.

## Hint 4: Shapes must chain
Each layer's output feeds the next: `Linear(1, 32)` then `Linear(32, 16)` then
`Linear(16, 1)`. The `out` of one Linear is the `in` of the next.

## Hint 5: `predict` = eval + no_grad
```python
model.eval()
with torch.no_grad():
    return model(x)
```

## Hint 6: Why the no_grad matters
If you return `model(x)` without wrapping it in `torch.no_grad()`, the output
still has `requires_grad=True` (because the parameters do), and the
`test_predict_has_no_grad` check fails. The `no_grad()` block detaches it.

## Hint 7: Parameter-count test failing?
Check your layer sizes: `Linear(1, 32)`, `Linear(32, 16)`, `Linear(16, 1)`. The
counts are `64 + 528 + 17 = 609`. A wrong dimension changes the total.
