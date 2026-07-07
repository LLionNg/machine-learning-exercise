# Learning: Pydantic v2 for ML configuration

[Pydantic](https://docs.pydantic.dev/latest/) turns a plain Python class into a
**data-validation schema**. You declare fields with type hints; Pydantic parses,
coerces, and validates any input against them and raises a clear
`ValidationError` when something is wrong. In ML code it's the standard way to
validate training configs, hyperparameters, and inference request/response
payloads.

> This challenge uses **Pydantic v2** (`pip show pydantic` -> 2.x). Some v1 APIs
> (`@validator`, `.dict()`, `class Config`) are renamed or removed in v2.

## 1. A basic model

```python
from pydantic import BaseModel

class LayerConfig(BaseModel):
    units: int
    activation: str = "relu"   # a default makes the field optional
    dropout: float = 0.0

LayerConfig(units=64)                       # ok
LayerConfig(units="64")                     # ok -> coerced to int 64
LayerConfig(units="abc")                    # raises ValidationError
```

## 2. Field constraints

Use [`Field`](https://docs.pydantic.dev/latest/concepts/fields/) to attach
validation rules to a field. Numeric bounds and length limits are declarative -
you don't write any code:

```python
from pydantic import BaseModel, Field

class Model(BaseModel):
    units: int = Field(gt=0)                 # > 0
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)   # 0.0 <= x <= 1.0
    name: str = Field(min_length=1, max_length=64)
    layers: list[int] = Field(min_length=1)  # non-empty list
```

| Constraint | Meaning |
|---|---|
| `gt`, `ge` | greater than / greater than or equal |
| `lt`, `le` | less than / less than or equal |
| `min_length`, `max_length` | for `str` **and** collections like `list` |
| `multiple_of` | numeric divisibility |

An equivalent, increasingly popular style uses `Annotated`:

```python
from typing import Annotated
units: Annotated[int, Field(gt=0)]
```

## 3. Restricting to a set of values with `Literal`

For "one of these exact strings" (optimizer names, activations, device), a
`Literal` is cleaner than a validator and gives great error messages:

```python
from typing import Literal

class Config(BaseModel):
    optimizer: Literal["sgd", "adam", "adamw"] = "adam"
    device: Literal["cpu", "cuda"] = "cpu"

Config(optimizer="rmsprop")   # raises ValidationError
```

(A `str`-based `enum.Enum` also works and is handy when you want methods on the
values.)

## 4. Nested models

A field typed as another `BaseModel` validates recursively. Pydantic even builds
the nested objects from plain dicts:

```python
class TrainingConfig(BaseModel):
    layers: list[LayerConfig] = Field(min_length=1)

cfg = TrainingConfig(layers=[{"units": 128}, {"units": 10, "activation": "none"}])
type(cfg.layers[0])   # -> LayerConfig, not dict
```

## 5. Custom validators

When a rule needs logic, or spans several fields, write a validator.

### `field_validator` - one field

Runs for a single field. `mode="before"` runs on the **raw** input (before type
coercion and `Field` constraints); `mode="after"` runs on the parsed value.
Normalising input (stripping whitespace) is a classic `before` use:

```python
from pydantic import field_validator

class TrainingConfig(BaseModel):
    model_name: str = Field(min_length=1, max_length=64)

    @field_validator("model_name", mode="before")
    @classmethod
    def strip_name(cls, value):
        # runs first, so "  resnet  " -> "resnet" and "   " -> "" (then
        # min_length=1 rejects it for you)
        return value.strip() if isinstance(value, str) else value
```

Note the `@classmethod` and that you **return** the (possibly transformed)
value.

### `model_validator` - the whole model

Runs once for the whole object. `mode="after"` gives you the fully-built
instance (`self`), which is perfect for **cross-field** rules:

```python
from pydantic import model_validator

class TrainingConfig(BaseModel):
    epochs: int = Field(gt=0)
    warmup_epochs: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_warmup(self):
        if self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs must not exceed epochs")
        return self          # after-validators return self
```

Raising `ValueError` (or `AssertionError`) inside any validator makes Pydantic
raise a `ValidationError` with your message attached.

## 6. Serialization

Once validated, get plain data back out - useful for logging a config or
writing it to disk:

```python
cfg.model_dump()          # -> dict
cfg.model_dump_json()     # -> JSON string
TrainingConfig.model_validate(some_dict)   # parse/validate a dict back into a model
```

## The "catch it early" mindset

The whole point is to **fail loudly at the boundary**. Instead of a mysterious
CUDA error 40 minutes into training because `learning_rate` was a string, the
config is rejected at construction with a message that names the field and the
rule it broke. That is exactly what the tests check: valid configs build and
expose the right values; invalid ones raise `ValidationError`.

## References

- Pydantic - Models: https://docs.pydantic.dev/latest/concepts/models/
- Pydantic - Fields: https://docs.pydantic.dev/latest/concepts/fields/
- Pydantic - Validators: https://docs.pydantic.dev/latest/concepts/validators/
