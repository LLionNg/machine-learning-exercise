from __future__ import annotations

import os

# Quiet TensorFlow's C++ logging (must be set before tensorflow is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf


def init_params(seed: int = 0) -> list[tf.Variable]:
    # seed tf, then create the six trainable variables [w1, b1, w2, b2, w3, b3]
    # for a [1, 32, 16, 1] MLP (see README for shapes)
    # TODO
    raise NotImplementedError


def forward(params: list[tf.Variable], x: tf.Tensor) -> tf.Tensor:
    # two hidden layers with sigmoid, then a linear output; maps (N, 1) to (N, 1)
    # TODO
    raise NotImplementedError


def mse(pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    # mean squared error: the mean of (pred - target) squared
    # TODO
    raise NotImplementedError


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
    raise NotImplementedError
