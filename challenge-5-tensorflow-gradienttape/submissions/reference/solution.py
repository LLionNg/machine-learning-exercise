from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf


def init_params(seed: int = 0) -> list[tf.Variable]:
    tf.random.set_seed(seed)
    w1 = tf.Variable(tf.random.normal((1, 32)))
    b1 = tf.Variable(tf.random.normal((32,)))
    w2 = tf.Variable(tf.random.normal((32, 16)))
    b2 = tf.Variable(tf.random.normal((16,)))
    w3 = tf.Variable(tf.random.normal((16, 1)))
    b3 = tf.Variable(tf.random.normal((1,)))
    return [w1, b1, w2, b2, w3, b3]


def forward(params: list[tf.Variable], x: tf.Tensor) -> tf.Tensor:
    w1, b1, w2, b2, w3, b3 = params
    x = tf.sigmoid(x @ w1 + b1)
    x = tf.sigmoid(x @ w2 + b2)
    return x @ w3 + b3


def mse(pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    e = pred - target
    return tf.reduce_mean(e * e)


def train(
    params: list[tf.Variable],
    x: tf.Tensor,
    y: tf.Tensor,
    epochs: int = 1000,
    lr: float = 0.1,
) -> list[float]:
    history: list[float] = []
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            loss = mse(forward(params, x), y)
        grads = tape.gradient(loss, params)
        for v, g in zip(params, grads):
            v.assign_sub(g * lr)
        history.append(float(loss.numpy()))
    return history
