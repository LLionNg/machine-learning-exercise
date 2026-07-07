# Hints - Challenge 1

Reveal these one at a time - try each step before the next hint.

## Hint 1: Constraints, not `if` statements
Most rules are a single `Field(...)` call. `units > 0` is `Field(gt=0)`;
`0.0 <= dropout <= 1.0` is `Field(ge=0.0, le=1.0)`; a non-empty list is
`Field(min_length=1)`.

## Hint 2: "One of these strings" -> `Literal`
Type `activation`, `optimizer`, and `device` as `Literal[...]` with the allowed
values. You get validation and a default in one line:

```python
optimizer: Literal["sgd", "adam", "adamw"] = "adam"
```

## Hint 3: Length uses `Field`, stripping needs a validator
Add `Field(min_length=1, max_length=64)` for the length, **and** a
`@field_validator("model_name", mode="before")` that returns `value.strip()`.
Because it runs *before* the length check, `"   "` becomes `""` and is rejected
by `min_length=1` automatically - you don't check for blank yourself.

## Hint 4: Don't forget `@classmethod`
A `field_validator` is a classmethod:

```python
@field_validator("model_name", mode="before")
@classmethod
def _strip(cls, value):
    return value.strip() if isinstance(value, str) else value
```

Guard with `isinstance(value, str)` so non-string input still reaches the normal
type error instead of crashing on `.strip()`.

## Hint 5: Cross-field rules use `model_validator`
`warmup_epochs <= epochs` compares two fields, so a single-field validator can't
see both. Use `@model_validator(mode="after")`, which receives the built instance
as `self`; compare, `raise ValueError(...)` if invalid, and `return self`.

## Hint 6: Nested models are automatic
Typing `layers: list[LayerConfig]` is all you need - Pydantic converts each dict
in the list into a `LayerConfig` and validates it. No manual loop.

## Hint 7: Let Pydantic raise
You never construct `ValidationError` yourself. A failed `Field` constraint or a
`raise ValueError(...)` inside a validator is automatically wrapped into a
`ValidationError`.

## Hint 8: Still stuck?
Run `pytest -v` and read the first failing assertion - the test name (e.g.
`test_layer_dropout_out_of_range`) tells you exactly which rule is missing.
