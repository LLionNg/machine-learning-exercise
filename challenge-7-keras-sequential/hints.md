# Hints - Challenge 7

Reveal these one at a time - try each step before the next hint.

## Hint 1: Sequential takes a list of layers
Pass a Python list to `keras.Sequential([...])`. Start with the input, then the
Dense layers:

```python
keras.Sequential([
    keras.Input(shape=(1,)),
    Dense(32, activation="sigmoid"),
    ...
])
```

## Hint 2: Activation goes inside Dense
Unlike PyTorch, the activation is an argument to the layer:
`Dense(32, activation="sigmoid")`. You don't add a separate activation layer.

## Hint 3: Dense infers its input size
You only give `Dense` the number of **output** units. `Dense(16, ...)` after a
`Dense(32, ...)` automatically takes 32 inputs. Only the `keras.Input(shape=(1,))`
declares the very first input size.

## Hint 4: The output layer is linear
`Dense(1)` with no `activation` argument. A regression output must be able to be
negative, so no sigmoid at the end.

## Hint 5: compile takes loss and optimizer
```python
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr))
```
Construct `SGD(lr)` explicitly (not the string `"sgd"`) so the learning rate is
actually set. `compile` returns nothing - it just configures the model.

## Hint 6: fit runs the loop and returns a History
```python
history = model.fit(x, y, epochs=epochs, verbose=0)
```
`verbose=0` keeps it quiet. The per-epoch losses live in
`history.history["loss"]`.

## Hint 7: Return a list of floats
`history.history["loss"]` may contain numpy floats. The tests expect a
`list[float]`, so convert: `[float(v) for v in history.history["loss"]]`.

## Hint 8: Loss/optimizer test failing?
Make sure you pass `loss="mse"` and `optimizer=keras.optimizers.SGD(lr)` to
`compile`, and that `compile_model` actually calls `model.compile(...)` (it's easy
to build the arguments but forget to call it).
