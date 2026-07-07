# Hints - Challenge 3

Reveal these one at a time - try each step before the next hint.

## Hint 1: `super().__init__()` first
The very first line of `MLP.__init__` must be `super().__init__()`, or PyTorch
can't register your layers and `model.parameters()` will be empty.

## Hint 2: Create layers as attributes
Assign each layer to `self.` so it's tracked:

```python
self.fc1 = nn.Linear(input_dim, hidden_dim)
self.relu = nn.ReLU()
self.fc2 = nn.Linear(hidden_dim, output_dim)
```

(Or wrap them in one `nn.Sequential` - either works.)

## Hint 3: `forward` returns logits
Chain the layers and return the result - **no sigmoid** here:
`return self.fc2(self.relu(self.fc1(x)))`. The output shape should be
`(N, output_dim)`.

## Hint 4: The training loop is 5 lines
In this order:

```python
optimizer.zero_grad()
loss = loss_fn(model(X), y)
loss.backward()
optimizer.step()
history.append(loss.item())
```

Forgetting `zero_grad()` makes gradients accumulate and training misbehave.

## Hint 5: Right loss, right dtypes
`nn.BCEWithLogitsLoss()` takes **raw logits** (don't sigmoid before it) and
**float** targets - `y` is already `float` of shape `(N, 1)`, matching the
logits' shape.

## Hint 6: Return Python floats
`.item()` converts a 0-dim tensor to a Python float. The tests expect `train(...)`
to return `list[float]`, so append `loss.item()`, not `loss`.

## Hint 7: Build the optimizer once
Create `loss_fn` and `optimizer` **before** the loop. Building a fresh `Adam`
optimizer inside the loop resets its internal state every step, so the loss won't
reliably go down.

## Hint 8: Forward-shape test failing?
Your layer dimensions are off - the first `Linear` must be
`Linear(input_dim, hidden_dim)` and the second `Linear(hidden_dim, output_dim)`,
with the `ReLU` *between* them.

## Hint 9: Gradient-flow test failing?
You most likely forgot `super().__init__()`, didn't assign layers to `self.` (so
they're not registered), or returned something detached from `forward`.

## Hint 10: Loss not going down?
Make sure you call `zero_grad()` each step, return **logits** (not a sigmoid) from
`forward`, and pass those logits straight into `BCEWithLogitsLoss`.
