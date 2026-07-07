# Challenge 7: Building and Training a Model with keras.Sequential

**Difficulty:** Beginner | **Est. time:** 30-40 min | **Topic:** `keras.Sequential`, `compile`, `fit`

This is the **Keras counterpart of Challenge 6**. You build the same regression
MLP as a `keras.Sequential`, but here you also get Keras's training workflow:
instead of writing the loop by hand (Challenges 4 and 5), you **configure** it
with `compile` and **run** it with `fit`.

That two-step workflow - `compile` then `fit` - is the thing that's genuinely new.
`compile` attaches a loss and an optimizer to the model; `fit` runs the whole
training loop for you and hands back a history of the loss.

## What to implement

### `build_model() -> keras.Model`

Return a `[1, 32, 16, 1]` regression MLP as a `keras.Sequential`. The layers:

| Position | Layer | Notes |
|---|---|---|
| 1 | `keras.Input(shape=(1,))` | declares the input is one feature |
| 2 | `Dense(32, activation="sigmoid")` | first hidden layer |
| 3 | `Dense(16, activation="sigmoid")` | second hidden layer |
| 4 | `Dense(1)` | output, **no** activation (linear) |

A `Dense(units)` layer is fully connected; it infers its input size from the
previous layer. The output `Dense(1)` has no activation so a regression output can
be any real number. The model has **609** parameters.

### `compile_model(model, lr=0.1) -> None`

Configure the model for training with `model.compile(...)`:

- **loss**: mean squared error (`"mse"`)
- **optimizer**: `keras.optimizers.SGD` with learning rate `lr`

`compile` mutates the model in place, so this function returns nothing.

### `train(model, x, y, epochs=1000) -> list[float]`

Train the already-compiled model with `model.fit(x, y, epochs=epochs, verbose=0)`.
`fit` returns a `History` object; return the per-epoch training loss
(`history.history["loss"]`) as a list of Python floats.

## How the tests check your work

1. **Structure** - it's a `keras.Model` with three `Dense` layers of units
   `32, 16, 1`; the first two use `sigmoid` and the output is linear; 609
   parameters.
2. **Data flows through (forward)** - calling the model on `(N, 1)` gives `(N, 1)`
   output.
3. **`compile` is configured** - after `compile_model`, the model's loss is MSE
   and its optimizer is `SGD` with learning rate `0.1`.
4. **`fit` learns** - training for ~120 epochs returns a loss history that goes
   **down** and never becomes `NaN`. (As elsewhere, we check the loss *decreases*,
   not that it converges.)

## Run the tests

```bash
uv sync                                # from the repo root, once: installs tensorflow + pytest
mkdir -p submissions/<your-username>
cp solution_template.py submissions/<your-username>/solution.py
# ...implement it, then:
./run_tests.sh                         # enter your username when prompted
```

See `learning.md` for the compile/fit primer and `hints.md` if you get stuck.
