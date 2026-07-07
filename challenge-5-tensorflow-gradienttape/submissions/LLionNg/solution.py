from __future__ import annotations

import os

# Quiet TensorFlow's C++ logging (must be set before tensorflow is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf


def init_params(seed: int = 0) -> list[tf.Variable]:
    # seed tf, then create the six trainable variables [w1, b1, w2, b2, w3, b3]
    # for a [1, 32, 16, 1] MLP (see README for shapes)
    # TODO
    tf.random.set_seed(seed)
    w1 = tf.Variable(tf.random.normal((1, 32)))
    b1 = tf.Variable(tf.random.normal((32,)))
    w2 = tf.Variable(tf.random.normal((32, 16)))
    b2 = tf.Variable(tf.random.normal((16,)))
    w3 = tf.Variable(tf.random.normal((16, 1)))
    b3 = tf.Variable(tf.random.normal((1,)))
    return [w1, b1, w2, b2, w3, b3]


def forward(params: list[tf.Variable], x: tf.Tensor) -> tf.Tensor:
    # two hidden layers with sigmoid, then a linear output; maps (N, 1) to (N, 1)
    # TODO
    w1, b1, w2, b2, w3, b3 = params
    x = tf.sigmoid(x @ w1 + b1)   # (N,1) @ (1,32) -> (N,32), + bias, squashed
    x = tf.sigmoid(x @ w2 + b2)   # (N,32) @ (32,16) -> (N,16)
    return x @ w3 + b3            # (N,16) @ (16,1) -> (N,1), LINEAR output


def mse(pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    # mean squared error: the mean of (pred - target) squared
    # TODO
    e = pred - target
    return tf.reduce_mean(e * e)   # a scalar tensor


def train(
    params: list[tf.Variable],
    x: tf.Tensor,
    y: tf.Tensor,
    epochs: int = 1000,
    lr: float = 0.1,
) -> list[float]:
    # manual gradient descent with GradientTape; each epoch: record the loss, get
    # grads via tape.gradient, update each variable with assign_sub. Return the
    # per-epoch loss as floats
    # TODO
    history = []
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            loss = mse(forward(params, x), y)   # forward, recorded
        grads = tape.gradient(loss, params)     # backward
        for v, g in zip(params, grads):
            v.assign_sub(g * lr)                # update
        history.append(float(loss.numpy()))     # .numpy() -> Python float
    return history
