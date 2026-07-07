import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

from solution import build_model, compile_model, train


# A tiny sin-wave regression fixture. Deterministic via a fixed seed.
def make_data(n: int = 100):
    tf.random.set_seed(1)
    x = tf.random.uniform((n, 1))
    y = tf.sin(3 * x + 1) + 0.5 * tf.random.uniform((n, 1))
    return x, y


# The model: it's a keras Sequential with the right layers
def test_returns_keras_model():
    assert isinstance(build_model(), keras.Model)


def test_layer_structure():
    layers = build_model().layers
    denses = [l for l in layers if isinstance(l, Dense)]
    assert len(denses) == 3, "expected three Dense layers"
    assert [l.units for l in denses] == [32, 16, 1]


def test_activations():
    denses = [l for l in build_model().layers if isinstance(l, Dense)]
    assert [l.activation.__name__ for l in denses] == ["sigmoid", "sigmoid", "linear"], (
        "the two hidden Dense layers use sigmoid; the output Dense has no activation (linear)"
    )


def test_parameter_count():
    # (1*32 + 32) + (32*16 + 16) + (16*1 + 1) = 64 + 528 + 17 = 609
    assert build_model().count_params() == 609


# Data flows through (forward)
def test_forward_output_shape():
    x, _ = make_data()
    assert tuple(build_model()(x).shape) == (100, 1), "forward should map (N, 1) -> (N, 1)"


# compile wires up the loss and optimizer
def test_compile_sets_loss_and_optimizer():
    model = build_model()
    compile_model(model, lr=0.1)
    loss = model.loss
    loss_name = loss if isinstance(loss, str) else loss.__class__.__name__
    assert loss_name.lower() in ("mse", "mean_squared_error", "meansquarederror"), "loss should be MSE"
    assert model.optimizer.__class__.__name__ == "SGD", "optimizer should be SGD"
    assert abs(float(model.optimizer.learning_rate) - 0.1) < 1e-4, "learning rate should be 0.1"


# fit trains the model
def test_train_returns_loss_history():
    x, y = make_data()
    model = build_model()
    compile_model(model)
    history = train(model, x, y, epochs=30)
    assert isinstance(history, list)
    assert len(history) == 30
    assert all(isinstance(v, float) for v in history)


def test_train_reduces_loss():
    x, y = make_data()
    model = build_model()
    compile_model(model, lr=0.1)
    history = train(model, x, y, epochs=120)
    assert all(v == v for v in history), "loss must never be NaN"  # NaN != NaN
    assert history[-1] < history[0], "model.fit should reduce the loss"
