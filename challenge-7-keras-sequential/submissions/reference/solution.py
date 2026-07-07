from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from tensorflow import keras
from tensorflow.keras.layers import Dense


def build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(1,)),
            Dense(32, activation="sigmoid"),
            Dense(16, activation="sigmoid"),
            Dense(1),
        ]
    )


def compile_model(model: keras.Model, lr: float = 0.1) -> None:
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr))


def train(model: keras.Model, x, y, epochs: int = 1000) -> list[float]:
    history = model.fit(x, y, epochs=epochs, verbose=0)
    return [float(v) for v in history.history["loss"]]
