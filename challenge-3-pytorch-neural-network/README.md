# Challenge 3: A Small Neural Network in PyTorch

**Difficulty:** Beginner-Intermediate | **Est. time:** 40-50 min | **Topic:** PyTorch `nn.Module`, forward pass, training loop

This challenge is about **writing a model**. You'll implement a tiny multilayer
perceptron (MLP) and confirm it's wired up correctly - data flows through the
forward pass and gradients flow back - plus a short training loop so you know how
a network learns.

Crucially, the tests **do not train the model to convergence**. They check that
the network is built correctly and that a *few* updates reduce the loss. So it's
fast and **CPU-only** (4 samples, 33 parameters) - no GPU, no long training.

## What to implement

### `MLP(nn.Module)` - the network *(the main task)*

Exactly this architecture - each row is one step of the forward pass:

| Step | Layer | Output shape |
|---|---|---|
| input | `x` | `(N, input_dim)` |
| 1 | `nn.Linear(input_dim, hidden_dim)` | `(N, hidden_dim)` |
| 2 | `nn.ReLU()` | `(N, hidden_dim)` |
| 3 | `nn.Linear(hidden_dim, output_dim)` | `(N, output_dim)`, the logits |

Defaults: `input_dim=2`, `hidden_dim=8`, `output_dim=1`. `forward(x)` returns the
raw **logits** (no sigmoid) of shape `(batch_size, output_dim)`.

With the defaults, the network has **33 parameters**: `(2*8 + 8) + (8*1 + 1)`.

### `train(model, X, y, epochs=20, lr=0.1) -> list[float]`

A standard full-batch training loop:

- loss: `nn.BCEWithLogitsLoss()` (expects raw logits and float targets)
- optimizer: `torch.optim.Adam(model.parameters(), lr=lr)` - **create it once**,
  before the loop
- each epoch: `zero_grad()` -> forward -> compute loss -> `backward()` -> `step()`

Return the per-epoch loss as a list of Python floats (`len == epochs`).

## How the tests check "does the network work?"

Rather than training for hundreds of epochs, the grader verifies the network the
way you'd sanity-check one in practice - cheaply:

1. **Structure** - two `nn.Linear` layers with the right shapes, a `ReLU`, and 33
   parameters.
2. **Data flows through (forward)** - feeding `(N, 2)` gives finite logits of
   shape `(N, 1)`; a single sample and custom dims also produce the right shape.
   *This is the "does the data go through and come out correctly?" check.*
3. **Gradients flow back (autograd)** - after one `loss.backward()`, every
   parameter has a non-`None`, non-zero gradient. A broken `forward` (wrong
   shapes, a detached graph, a missing layer) fails here immediately.
4. **The loop learns a little** - over ~20 updates the loss goes **down** (and
   never becomes `NaN`). This confirms the whole loop is wired up, without
   requiring the model to fully solve the task.

Sample data is the classic XOR batch, but you are **not** asked to drive it to
100% accuracy - just to build a correct network and a working training step.

## Run the tests

```bash
uv sync                                # from the repo root, once: installs torch + pytest
mkdir -p submissions/<your-username>
cp solution_template.py submissions/<your-username>/solution.py
# ...implement it, then:
./run_tests.sh                         # enter your username when prompted
```

See `learning.md` for a PyTorch primer and `hints.md` if you get stuck.
