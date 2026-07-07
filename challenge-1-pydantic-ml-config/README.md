# Challenge 1: Pydantic Models for an ML Training Config

**Difficulty:** Beginner | **Est. time:** 30-45 min | **Topic:** Pydantic v2 data validation

Configuration bugs are one of the most common causes of wasted GPU hours: a
negative learning rate, a typo in the optimizer name, a dropout of `1.5`. In
production ML systems, [Pydantic](https://docs.pydantic.dev/latest/) is the
standard tool for catching these mistakes *before* training starts.

Your task is to implement two Pydantic v2 models that validate a training
configuration.

## What to implement

### `LayerConfig` - one dense layer

| Field | Type | Rule |
|---|---|---|
| `units` | `int` | must be **> 0** |
| `activation` | one of `"relu"`, `"tanh"`, `"sigmoid"`, `"none"` | default `"relu"` |
| `dropout` | `float` | inclusive range **[0.0, 1.0]**, default `0.0` |

### `TrainingConfig` - the whole run

| Field | Type | Rule |
|---|---|---|
| `model_name` | `str` | non-empty, <= 64 chars, **whitespace stripped** first |
| `learning_rate` | `float` | range **(0.0, 1.0]** |
| `batch_size` | `int` | **> 0**, <= 4096 |
| `epochs` | `int` | **> 0**, <= 1000 |
| `warmup_epochs` | `int` | **>= 0**, default `0`, and **must not exceed `epochs`** |
| `optimizer` | one of `"sgd"`, `"adam"`, `"adamw"` | default `"adam"` |
| `layers` | `list[LayerConfig]` | at least one layer |
| `device` | one of `"cpu"`, `"cuda"` | default `"cpu"` |
| `seed` | `int` or `None` | default `None` |

Two rules need custom validators:

- **`model_name` is stripped before validation** - `"  resnet  "` becomes
  `"resnet"`, and a name that is only whitespace is rejected. Use a
  `field_validator(mode="before")`.
- **`warmup_epochs` must not exceed `epochs`** - this compares two fields, so
  use a `model_validator(mode="after")` and raise `ValueError` when violated.

## Expected behaviour

```python
from solution import TrainingConfig

cfg = TrainingConfig(
    model_name="  my-net  ",
    learning_rate=0.001,
    batch_size=64,
    epochs=20,
    layers=[{"units": 128, "activation": "relu", "dropout": 0.2}, {"units": 10}],
)
cfg.model_name          # -> "my-net"   (stripped)
cfg.optimizer           # -> "adam"     (default)
cfg.layers[1].activation  # -> "relu"   (default on the nested model)

TrainingConfig(model_name="x", learning_rate=5.0, batch_size=64, epochs=1,
               layers=[{"units": 8}])
# -> raises pydantic.ValidationError  (learning_rate must be <= 1.0)
```

Invalid input must raise `pydantic.ValidationError` (Pydantic does this for you
when a `Field` constraint fails or a validator raises `ValueError`).

## Run the tests

```bash
uv sync                 # from the repo root, once: installs deps into .venv
# put your solution where the runner expects it:
mkdir -p submissions/<your-username>
cp solution_template.py submissions/<your-username>/solution.py
# ...implement it, then:
./run_tests.sh          # enter your username when prompted
```

Or directly: copy your file to `solution.py` beside `test_solution.py` and run
`uv run pytest -v`.

See `learning.md` for a Pydantic v2 primer and `hints.md` if you get stuck.
