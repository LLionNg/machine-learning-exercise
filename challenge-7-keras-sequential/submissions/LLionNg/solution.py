from __future__ import annotations

import os

# Quiet TensorFlow's C++ logging (must be set before tensorflow is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from tensorflow import keras
from tensorflow.keras.layers import Dense

def build_model() -> keras.Model:
    # a [1, 32, 16, 1] regression MLP as a keras.Sequential: an Input of shape
    # (1,), Dense(32) and Dense(16) with sigmoid activation, then a Dense(1)
    # output with no activation
    # TODO
    model = keras.Sequential([
        keras.Input(shape=(1,)),        # one input feature
        Dense(32, activation="sigmoid"),
        Dense(16, activation="sigmoid"),
        Dense(1),                        # linear output
    ])
    return model


def compile_model(model: keras.Model, lr: float = 0.1) -> None:
    # configure training: mean-squared-error loss and an SGD optimizer with the
    # given learning rate (compile mutates the model in place; return nothing)
    # TODO
    model.compile(
        loss="mse",                              # mean squared error
        optimizer=keras.optimizers.SGD(0.1)     # stochastic gradient descent, lr=0.1
    )

def train(model: keras.Model, x, y, epochs: int = 1000) -> list[float]:
    # fit the (already compiled) model for `epochs`, silently, and return the
    # per-epoch training loss as a list of floats
    # TODO
    history = model.fit(x, y, epochs=epochs, verbose=0)
    return [float(v) for v in history.history["loss"]]
