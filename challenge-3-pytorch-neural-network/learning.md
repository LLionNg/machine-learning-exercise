# Learning: Building and training a neural network in PyTorch

This is a tour of the PyTorch pieces you need for this challenge. It follows the
official ["Learn the Basics"](https://pytorch.org/tutorials/beginner/basics/intro.html)
tutorial, trimmed to the essentials.

## 1. Tensors

A `torch.Tensor` is an n-dimensional array (like NumPy) that can also track
gradients. Our data is tiny:

```python
import torch
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # shape (4, 2)
y = torch.tensor([[0.], [1.], [1.], [0.]])                   # shape (4, 1)
```

Shapes matter: the network maps `(N, 2) -> (N, 1)`.

## 2. Defining a network: subclass `nn.Module`

Every model subclasses [`nn.Module`](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).
You create the layers in `__init__` (after calling `super().__init__()`), and
describe the forward pass in `forward`:

```python
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
        super().__init__()                       # required, and first
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)                        # raw logits
```

- **`nn.Linear(in, out)`** is a fully-connected layer: `y = x @ W^T + b`. It
  registers a weight matrix of shape `(out, in)` and a bias of shape `(out,)`.
- **`nn.ReLU()`** is the non-linearity `max(0, x)`. Without a non-linearity
  between the two Linear layers, the whole network collapses into a single linear
  map - and a linear map **cannot** solve XOR.
- Call the model as `model(x)`, **not** `model.forward(x)` - `__call__` runs
  hooks and then `forward`.

An equivalent style uses `nn.Sequential` to chain modules:

```python
self.net = nn.Sequential(
    nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
)
# forward: return self.net(x)
```

Either style passes the tests, as long as the layer shapes are correct.

### Logits vs. probabilities

`forward` returns **logits** - unbounded real numbers. You convert them to a
probability with `sigmoid` only when you need a 0/1 decision. Keeping the model's
output as logits is what lets us use the numerically-stable loss below.

### Counting parameters

```python
sum(p.numel() for p in model.parameters())
```

For our defaults: `fc1` has `2*8 + 8 = 24`, `fc2` has `8*1 + 1 = 9`, total **33**.

## 3. Loss and optimizer

For binary classification with a single logit, use
**`nn.BCEWithLogitsLoss`** - it combines a sigmoid and binary cross-entropy in one
numerically-stable step, so you feed it **raw logits** and **float** targets
(`0.0`/`1.0`).

```python
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
```

> If you instead had 2+ output classes, you'd use `nn.CrossEntropyLoss` with an
> `output_dim` equal to the number of classes and integer (`long`) targets. Here
> a single logit + `BCEWithLogitsLoss` is the simplest fit.

`Adam` is a solid default optimizer. `SGD` also works but usually needs more
epochs on XOR.

## 4. The training loop

The three lines that actually learn are `zero_grad()`, `backward()`, `step()`:

```python
model.train()
history = []
for _ in range(epochs):
    optimizer.zero_grad()      # clear gradients from the previous step
    logits = model(X)          # forward pass
    loss = loss_fn(logits, y)  # compare to targets
    loss.backward()            # backprop: fill each parameter's .grad
    optimizer.step()           # nudge parameters down the gradient
    history.append(loss.item())  # .item() -> Python float
```

Why `zero_grad()`? PyTorch **accumulates** gradients by default, so you must
reset them each step or they'd sum across iterations.

Because our dataset is only 4 rows, we train on the full batch every epoch - no
`DataLoader` needed.

## 5. How to know a network "works" - without training it to convergence

You don't need to train for hundreds of epochs to know a network is built
correctly. The fast, standard checks are:

- **Forward shape & finiteness.** Push a batch through and confirm the output has
  the expected shape and contains no `NaN`/`inf`. This is the "does the data go
  through and come out correctly?" check - it catches wrong layer dimensions and
  broken wiring instantly.
  ```python
  out = model(X)                 # X: (N, 2)
  assert out.shape == (X.shape[0], 1)
  assert torch.isfinite(out).all()
  ```
- **Gradients flow.** After a single `loss.backward()`, every parameter should
  have a non-`None`, non-zero `.grad`. If a layer isn't registered, or the graph
  was detached, or `forward` returns the wrong thing, gradients won't reach the
  parameters.
  ```python
  loss = nn.BCEWithLogitsLoss()(model(X), y)
  loss.backward()
  assert all(p.grad is not None for p in model.parameters())
  ```
- **A few steps reduce the loss.** Run the loop for a handful of epochs and check
  the loss goes down. You don't need it to reach zero - just to *move in the right
  direction*, which proves `zero_grad`/`backward`/`step` are all in place.

These three checks - which are exactly what this challenge's tests do - validate
the model in about a second on the CPU. (Fully training a network to solve a task
is the subject of a later challenge.)

## 6. (For later) Inference

Once a model *is* trained, you turn logits into decisions without tracking
gradients:

```python
with torch.no_grad():          # disables autograd -> faster, less memory
    model.eval()               # switches layers like dropout/batchnorm to eval mode
    probs = torch.sigmoid(model(X))
    preds = (probs >= 0.5).float()
```

You don't need this for the challenge, but it's the natural next step.

## References

- Build the model: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- Optimization / training loop: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- `torch.nn` API: https://pytorch.org/docs/stable/nn.html
