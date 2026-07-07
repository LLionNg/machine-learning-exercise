# Learning: composing models with nn.Sequential

This follows the "Build the Neural Network" section of PyTorch's
["Learn the Basics"](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
tutorial and the `nn.Sequential`
[API docs](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html).

## 1. Two ways to define the same network

In Challenge 3 you subclassed `nn.Module`:

```python
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)
```

When the data just flows straight through, one layer after another, that class is
boilerplate. `nn.Sequential` is a container that chains modules for you and
supplies the `forward` automatically:

```python
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Sigmoid(),
    nn.Linear(32, 16),
    nn.Sigmoid(),
    nn.Linear(16, 1),
)
```

Calling `model(x)` runs each layer in order and returns the last one's output. The
two are equivalent networks - Sequential is just less typing when there's no
branching or custom logic.

**When to use which:** reach for `nn.Sequential` for a simple linear stack. Reach
for an `nn.Module` subclass when `forward` needs anything non-trivial - skip
connections, multiple inputs, branches, or reusing a layer.

## 2. Activations go *between* the Linear layers

A `nn.Linear` is an affine map. Stacking two of them with nothing in between is
still just one affine map, so the network could only ever learn straight lines.
The `nn.Sigmoid()` between them adds the non-linearity that lets the network bend
to fit a curve like `sin`.

The **last** layer has no activation. This is a regression - the target
`y = sin(3x + 1) + noise` takes negative and positive values, so the output must
be free to be any real number. A trailing `sigmoid` would clamp it to `(0, 1)`.

## 3. Counting parameters

Each `nn.Linear(in, out)` holds a weight of shape `(out, in)` and a bias of shape
`(out,)`, so `in*out + out` numbers:

```python
sum(p.numel() for p in model.parameters())
```

- `Linear(1, 32)`  -> `1*32 + 32   = 64`
- `Linear(32, 16)` -> `32*16 + 16  = 528`
- `Linear(16, 1)`  -> `16*1 + 1    = 17`
- total: **609**

The `Sigmoid` layers have no parameters.

## 4. Inference: eval mode and no_grad

Once you have a model, you often just want its predictions - no learning. Two
habits make inference correct and efficient:

```python
model.eval()                 # switch layers like Dropout/BatchNorm to eval behaviour
with torch.no_grad():        # stop tracking gradients: faster, less memory
    preds = model(x)
```

- **`model.eval()`** flips the module into evaluation mode. Our tiny model has no
  Dropout or BatchNorm, so it makes no numerical difference here - but it's the
  correct habit, because those layers behave differently during training vs.
  inference.
- **`torch.no_grad()`** tells autograd not to build the backward graph. The
  returned tensor is detached (`requires_grad` is `False`), which is exactly what
  you want for a prediction you're not going to backprop through.

If you return `model(x)` *without* `no_grad`, the output still carries
`requires_grad=True` (because the parameters do), needlessly holding onto the
graph.

## References

- Build the Neural Network: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- `nn.Sequential`: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
- `torch.no_grad`: https://pytorch.org/docs/stable/generated/torch.no_grad.html
