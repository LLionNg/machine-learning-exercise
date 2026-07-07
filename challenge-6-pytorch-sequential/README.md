# Challenge 6: Building a Model with torch.nn.Sequential

**Difficulty:** Beginner | **Est. time:** 25-35 min | **Topic:** `nn.Sequential`, layer composition, inference

Challenge 3 built a network by **subclassing** `nn.Module` and writing a `forward`
method. When a model is just a straight stack of layers - each feeding the next -
you don't need a class at all. `nn.Sequential` lets you compose it from a list.

This is a short challenge focused on two things: **composing** the model, and
**using** it for inference. There's no training loop here - Challenges 3 and 4
already cover that. The point is the container and how you run a finished model.

## What to implement

### `build_model() -> nn.Sequential`

Return a `[1, 32, 16, 1]` regression MLP as an `nn.Sequential`. The layers, in
order:

| Position | Layer | Output shape |
|---|---|---|
| 1 | `nn.Linear(1, 32)` | `(N, 32)` |
| 2 | `nn.Sigmoid()` | `(N, 32)` |
| 3 | `nn.Linear(32, 16)` | `(N, 16)` |
| 4 | `nn.Sigmoid()` | `(N, 16)` |
| 5 | `nn.Linear(16, 1)` | `(N, 1)` |

The two `Sigmoid` activations sit **between** the Linear layers to provide
non-linearity. The final layer is **linear** (no activation) so a regression
output can be any real number. With these sizes the model has **609**
parameters.

### `predict(model, x) -> torch.Tensor`

Run inference. A finished model used for prediction should not track gradients:

- put the model in evaluation mode with `model.eval()`
- compute the output inside a `torch.no_grad()` block
- return the predictions, shape `(N, 1)`

## How the tests check your work

1. **Structure** - it's an `nn.Sequential` with three `Linear` layers of the right
   shapes, two `Sigmoid` activations, and 609 parameters.
2. **Data flows through (forward)** - feeding `(N, 1)` gives finite `(N, 1)`
   output.
3. **Gradients flow** - one `MSELoss().backward()` gives every parameter a
   non-`None`, non-zero gradient, confirming the model is trainable.
4. **Inference is clean** - `predict` returns predictions of shape `(N, 1)` that
   are **detached from the graph** (`requires_grad` is `False`).

## Run the tests

```bash
uv sync                                # from the repo root, once
mkdir -p submissions/<your-username>
cp solution_template.py submissions/<your-username>/solution.py
# ...implement it, then:
./run_tests.sh                         # enter your username when prompted
```

See `learning.md` for a primer on `nn.Sequential` and inference, and `hints.md` if
you get stuck.
