# Learning: keras.Sequential, compile, and fit

This follows the Keras
["The Sequential model"](https://keras.io/guides/sequential_model/) guide and the
`compile`/`fit` workflow from
["Training & evaluation with the built-in methods"](https://keras.io/guides/training_with_built_in_methods/).

## 1. keras.Sequential: a stack of layers

Like `torch.nn.Sequential` in Challenge 6, `keras.Sequential` chains layers so you
don't write a forward pass:

```python
from tensorflow import keras
from tensorflow.keras.layers import Dense

model = keras.Sequential([
    keras.Input(shape=(1,)),        # one input feature
    Dense(32, activation="sigmoid"),
    Dense(16, activation="sigmoid"),
    Dense(1),                        # linear output
])
```

- **`keras.Input(shape=(1,))`** declares the input shape (a single feature). It
  lets Keras build the layers immediately, so `model.count_params()` works right
  away.
- **`Dense(units, activation=...)`** is a fully-connected layer - Keras's
  equivalent of `nn.Linear` plus an optional activation baked in. Unlike PyTorch,
  you don't pass the input size; Keras infers it from the previous layer.
- The activation lives **inside** the `Dense` here, rather than being a separate
  layer as in PyTorch. `Dense(32, activation="sigmoid")` is "linear then sigmoid".
- The final `Dense(1)` has **no** activation - a linear output, because a
  regression target can be any real number.

Parameters count the same as the PyTorch version: `64 + 528 + 17 = 609`.

## 2. compile: attach a loss and an optimizer

In Challenges 4 and 5 you wrote the training loop by hand. Keras takes a different
approach: you first **configure** training by attaching a loss and an optimizer to
the model.

```python
model.compile(
    loss="mse",                              # mean squared error
    optimizer=keras.optimizers.SGD(0.1)     # stochastic gradient descent, lr=0.1
)
```

- **`loss="mse"`** is the same mean-squared-error you computed by hand before -
  Keras just has a built-in name for it.
- **`keras.optimizers.SGD(0.1)`** is plain gradient descent with a learning rate
  of `0.1`. (If you wrote `optimizer="sgd"` you'd get the default learning rate
  and couldn't set it, which is why we construct `SGD(0.1)` explicitly.)

`compile` doesn't train anything - it just records *how* to train. It mutates the
model in place and returns `None`.

## 3. fit: run the whole training loop

`model.fit` runs the entire loop - forward, loss, gradients, weight update, repeat
- for as many epochs as you ask:

```python
history = model.fit(x, y, epochs=1000, verbose=0)
losses = history.history["loss"]     # one loss value per epoch
```

- **`epochs`** is how many passes over the data.
- **`verbose=0`** silences the progress bar (handy for tests and scripts).
- **`history.history["loss"]`** is a list with the training loss after each epoch
  - exactly the loss curve you built by hand in Challenges 4 and 5.

## 4. The same loop, two philosophies

You have now trained the same network three ways. Side by side:

| | You write... | The framework provides... |
|---|---|---|
| ch. 4 (PyTorch autograd) | every step by hand | just `backward()` |
| ch. 5 (TF GradientTape) | every step by hand | just the tape + gradients |
| ch. 7 (Keras compile/fit) | *what* to optimize | the entire loop (`fit`) |

`compile`/`fit` is the highest-level of the three: you declare the loss and
optimizer, and Keras runs everything. It's the fastest to write, at the cost of
seeing less of what happens inside - which is exactly why doing it by hand first
was worth it.

## 5. Verifying without training to convergence

Same idea as the other model challenges: we don't train to a perfect fit. We check
the model is built correctly (layers, activations, parameter count), that data
flows through it, that `compile` set the loss and optimizer, and that a short
`fit` run makes the loss go **down**.

## References

- The Sequential model: https://keras.io/guides/sequential_model/
- Training with built-in methods (compile/fit): https://keras.io/guides/training_with_built_in_methods/
- `Dense` layer: https://keras.io/api/layers/core_layers/dense/
