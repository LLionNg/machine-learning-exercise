import torch
from torch import nn

from solution import build_model, predict


# The model: it's a Sequential with the right layers
def test_returns_sequential():
    assert isinstance(build_model(), nn.Sequential)


def test_layer_structure():
    model = build_model()
    linears = [m for m in model if isinstance(m, nn.Linear)]
    assert len(linears) == 3, "expected three nn.Linear layers"
    assert (linears[0].in_features, linears[0].out_features) == (1, 32)
    assert (linears[1].in_features, linears[1].out_features) == (32, 16)
    assert (linears[2].in_features, linears[2].out_features) == (16, 1)
    assert sum(isinstance(m, nn.Sigmoid) for m in model) == 2, "expected two Sigmoid activations"


def test_parameter_count():
    # (1*32 + 32) + (32*16 + 16) + (16*1 + 1) = 64 + 528 + 17 = 609
    assert sum(p.numel() for p in build_model().parameters()) == 609


# Data flows through (forward)
def test_forward_output_shape_and_finite():
    out = build_model()(torch.rand(20, 1))
    assert out.shape == (20, 1), "forward should map (N, 1) -> (N, 1)"
    assert torch.isfinite(out).all(), "output must be finite (no NaN/inf)"


def test_forward_single_sample():
    assert build_model()(torch.tensor([[0.5]])).shape == (1, 1)


# Gradients flow back (the model is trainable)
def test_gradients_flow():
    model = build_model()
    x = torch.rand(20, 1)
    y = torch.rand(20, 1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads), "every parameter should receive a gradient"
    assert any(g.abs().sum().item() > 0 for g in grads), "gradients should be non-zero"


# Inference: predict runs without tracking gradients
def test_predict_shape():
    out = predict(build_model(), torch.rand(20, 1))
    assert out.shape == (20, 1)


def test_predict_has_no_grad():
    # Inference must not track gradients, even though the model's parameters do.
    out = predict(build_model(), torch.rand(20, 1))
    assert not out.requires_grad, "predict should return a tensor detached from the graph (use torch.no_grad)"
