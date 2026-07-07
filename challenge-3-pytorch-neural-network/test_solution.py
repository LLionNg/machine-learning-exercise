import torch
from torch import nn

from solution import MLP, train


# A tiny batch (the XOR inputs/targets). We use it only as sample data to push
# through the network -- the tests do not require the model to *solve* XOR.
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


# The model: structure
def test_is_nn_module():
    assert isinstance(MLP(), nn.Module)


def test_architecture_layers():
    model = MLP()
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2, "expected exactly two nn.Linear layers"
    assert (linears[0].in_features, linears[0].out_features) == (2, 8)
    assert (linears[1].in_features, linears[1].out_features) == (8, 1)
    assert any(isinstance(m, nn.ReLU) for m in model.modules()), "expected a ReLU activation"


def test_parameter_count():
    # (2*8 + 8) + (8*1 + 1) = 24 + 9 = 33
    assert sum(p.numel() for p in MLP().parameters()) == 33


# The model: data flows through (forward pass)
def test_forward_output_shape_and_finite():
    out = MLP()(X)
    assert out.shape == (4, 1), "forward should map (N, 2) -> (N, 1)"
    assert torch.isfinite(out).all(), "output must be finite (no NaN/inf)"


def test_forward_single_sample():
    out = MLP()(torch.tensor([[1.0, 0.0]]))
    assert out.shape == (1, 1)


def test_forward_respects_custom_dims():
    out = MLP(input_dim=3, hidden_dim=5, output_dim=2)(torch.rand(7, 3))
    assert out.shape == (7, 2)


# The model: gradients flow back (autograd is wired up)
def test_gradients_flow():
    model = MLP()
    loss = nn.BCEWithLogitsLoss()(model(X), y)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads), "every parameter should receive a gradient"
    assert any(g.abs().sum().item() > 0 for g in grads), "gradients should be non-zero"


# The training loop: mechanics (a few steps reduce the loss -- not convergence)
def test_train_returns_loss_history():
    torch.manual_seed(0)
    history = train(MLP(), X, y, epochs=10, lr=0.1)
    assert isinstance(history, list)
    assert len(history) == 10
    assert all(isinstance(v, float) for v in history)


def test_train_reduces_loss():
    torch.manual_seed(0)
    history = train(MLP(), X, y, epochs=20, lr=0.1)
    assert all(v == v for v in history), "loss must never be NaN"  # NaN != NaN
    assert history[-1] < history[0], "a few training steps should reduce the loss"
