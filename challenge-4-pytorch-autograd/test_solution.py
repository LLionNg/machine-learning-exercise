import torch

from solution import forward, init_params, mse, train


# A tiny sin-wave regression fixture (the same shape of problem the notebook
# used). Deterministic via a fixed seed.
def make_data(n: int = 100):
    torch.manual_seed(1)
    x = torch.rand(n, 1)
    y = torch.sin(3 * x + 1) + 0.5 * torch.rand(n, 1)
    return x, y


# init_params: the raw leaf tensors
def test_init_params_shapes():
    params = init_params()
    assert len(params) == 6, "expected [w1, b1, w2, b2, w3, b3]"
    shapes = [tuple(p.shape) for p in params]
    assert shapes == [(1, 32), (32,), (32, 16), (16,), (16, 1), (1,)]


def test_init_params_require_grad_and_are_leaves():
    for p in init_params():
        assert p.requires_grad, "every parameter must have requires_grad=True"
        assert p.is_leaf, "parameters must be leaf tensors (created directly, not from an op)"


# forward: data flows through
def test_forward_output_shape_and_finite():
    x, _ = make_data()
    out = forward(init_params(), x)
    assert out.shape == (100, 1), "forward should map (N, 1) -> (N, 1)"
    assert torch.isfinite(out).all(), "output must be finite (no NaN/inf)"


def test_forward_single_sample():
    out = forward(init_params(), torch.tensor([[0.5]]))
    assert out.shape == (1, 1)


# mse: exact value
def test_mse_known_value():
    pred = torch.tensor([[2.0], [4.0]])
    target = torch.tensor([[0.0], [0.0]])
    # mean(2^2, 4^2) = mean(4, 16) = 10
    assert abs(mse(pred, target).item() - 10.0) < 1e-6


# gradients flow back to the raw tensors
def test_gradients_flow():
    x, y = make_data()
    params = init_params()
    loss = mse(forward(params, x), y)
    loss.backward()
    grads = [p.grad for p in params]
    assert all(g is not None for g in grads), "every parameter should receive a gradient"
    assert any(g.abs().sum().item() > 0 for g in grads), "gradients should be non-zero"


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
    # After training, the parameters must have actually moved (proving the manual
    # update ran), and their gradients must be zeroed (proving grad.zero_()).
    x, y = make_data()
    params = init_params()
    before = params[0].detach().clone()
    train(params, x, y, epochs=20, lr=0.1)
    assert not torch.allclose(before, params[0].detach()), "weights should change during training"
    assert torch.count_nonzero(params[0].grad) == 0, "gradients should be zeroed each step"
