import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

from solution import forward, init_params, mse, train


# A tiny sin-wave regression fixture (the same shape of problem the notebook
# used). Deterministic via a fixed seed.
def make_data(n: int = 100):
    tf.random.set_seed(1)
    x = tf.random.uniform((n, 1))
    y = tf.sin(3 * x + 1) + 0.5 * tf.random.uniform((n, 1))
    return x, y


# init_params: the trainable variables
def test_init_params_shapes():
    params = init_params()
    assert len(params) == 6, "expected [w1, b1, w2, b2, w3, b3]"
    shapes = [tuple(p.shape) for p in params]
    assert shapes == [(1, 32), (32,), (32, 16), (16,), (16, 1), (1,)]


def test_init_params_are_trainable_variables():
    for p in init_params():
        assert isinstance(p, tf.Variable), "every parameter must be a tf.Variable"
        assert p.trainable, "variables must be trainable"


# forward: data flows through
def test_forward_output_shape_and_finite():
    x, _ = make_data()
    out = forward(init_params(), x)
    assert tuple(out.shape) == (100, 1), "forward should map (N, 1) -> (N, 1)"
    assert bool(tf.reduce_all(tf.math.is_finite(out))), "output must be finite (no NaN/inf)"


def test_forward_single_sample():
    out = forward(init_params(), tf.constant([[0.5]]))
    assert tuple(out.shape) == (1, 1)


# mse: exact value
def test_mse_known_value():
    pred = tf.constant([[2.0], [4.0]])
    target = tf.constant([[0.0], [0.0]])
    # mean(2^2, 4^2) = mean(4, 16) = 10
    assert abs(float(mse(pred, target).numpy()) - 10.0) < 1e-6


# gradients flow back to the variables through the tape
def test_gradients_flow():
    x, y = make_data()
    params = init_params()
    with tf.GradientTape() as tape:
        loss = mse(forward(params, x), y)
    grads = tape.gradient(loss, params)
    assert all(g is not None for g in grads), "every variable should receive a gradient"
    assert any(float(tf.reduce_sum(tf.abs(g))) > 0 for g in grads), "gradients should be non-zero"


# the manual training loop
def test_train_returns_loss_history():
    x, y = make_data()
    history = train(init_params(), x, y, epochs=50, lr=0.1)
    assert isinstance(history, list)
    assert len(history) == 50
    assert all(isinstance(v, float) for v in history)


def test_train_reduces_loss():
    x, y = make_data()
    history = train(init_params(), x, y, epochs=300, lr=0.1)
    assert all(v == v for v in history), "loss must never be NaN"  # NaN != NaN
    assert history[-1] < history[0], "manual gradient descent should reduce the loss"


def test_train_updates_the_weights():
    # After training the variables must have actually moved (proving assign_sub ran).
    x, y = make_data()
    params = init_params()
    before = tf.identity(params[0])
    train(params, x, y, epochs=20, lr=0.1)
    assert not bool(tf.reduce_all(before == params[0])), "weights should change during training"
