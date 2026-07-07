import pytest
from pydantic import ValidationError

from solution import LayerConfig, TrainingConfig


def make_config(**overrides):
    """A valid TrainingConfig kwargs dict, with per-test overrides applied."""
    base = dict(
        model_name="resnet",
        learning_rate=0.01,
        batch_size=32,
        epochs=10,
        layers=[{"units": 16, "activation": "relu", "dropout": 0.1}],
    )
    base.update(overrides)
    return base


# LayerConfig
def test_layer_minimal_defaults():
    layer = LayerConfig(units=8)
    assert layer.units == 8
    assert layer.activation == "relu"
    assert layer.dropout == 0.0


@pytest.mark.parametrize("bad_units", [0, -1, -100])
def test_layer_units_must_be_positive(bad_units):
    with pytest.raises(ValidationError):
        LayerConfig(units=bad_units)


def test_layer_activation_accepts_allowed_values():
    for act in ("relu", "tanh", "sigmoid", "none"):
        assert LayerConfig(units=4, activation=act).activation == act


def test_layer_activation_rejects_unknown():
    with pytest.raises(ValidationError):
        LayerConfig(units=4, activation="gelu")


@pytest.mark.parametrize("bad_dropout", [-0.1, 1.5, 2.0])
def test_layer_dropout_out_of_range(bad_dropout):
    with pytest.raises(ValidationError):
        LayerConfig(units=4, dropout=bad_dropout)


def test_layer_dropout_boundaries_ok():
    assert LayerConfig(units=4, dropout=0.0).dropout == 0.0
    assert LayerConfig(units=4, dropout=1.0).dropout == 1.0


# TrainingConfig
def test_training_valid_full():
    cfg = TrainingConfig(**make_config(
        warmup_epochs=2, optimizer="sgd", device="cuda", seed=123,
        layers=[{"units": 32}, {"units": 16, "activation": "tanh", "dropout": 0.5}],
    ))
    assert cfg.model_name == "resnet"
    assert cfg.optimizer == "sgd"
    assert cfg.device == "cuda"
    assert cfg.seed == 123
    assert len(cfg.layers) == 2
    assert cfg.layers[1].activation == "tanh"


def test_training_defaults():
    cfg = TrainingConfig(**make_config())
    assert cfg.optimizer == "adam"
    assert cfg.device == "cpu"
    assert cfg.warmup_epochs == 0
    assert cfg.seed is None


# TrainingConfig - numeric & literal constraints
@pytest.mark.parametrize("lr", [0.0, -0.1, 1.5])
def test_learning_rate_invalid(lr):
    with pytest.raises(ValidationError):
        TrainingConfig(**make_config(learning_rate=lr))


def test_learning_rate_upper_bound_ok():
    assert TrainingConfig(**make_config(learning_rate=1.0)).learning_rate == 1.0


@pytest.mark.parametrize("bs", [0, -1, 5000])
def test_batch_size_invalid(bs):
    with pytest.raises(ValidationError):
        TrainingConfig(**make_config(batch_size=bs))


@pytest.mark.parametrize("ep", [0, -3, 2000])
def test_epochs_invalid(ep):
    with pytest.raises(ValidationError):
        TrainingConfig(**make_config(epochs=ep))


def test_optimizer_literal():
    with pytest.raises(ValidationError):
        TrainingConfig(**make_config(optimizer="rmsprop"))


def test_device_literal():
    with pytest.raises(ValidationError):
        TrainingConfig(**make_config(device="tpu"))


def test_layers_must_be_non_empty():
    with pytest.raises(ValidationError):
        TrainingConfig(**make_config(layers=[]))


# TrainingConfig - custom validators
def test_model_name_is_stripped():
    cfg = TrainingConfig(**make_config(model_name="  my-model  "))
    assert cfg.model_name == "my-model"


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_model_name_blank_rejected(blank):
    with pytest.raises(ValidationError):
        TrainingConfig(**make_config(model_name=blank))


def test_model_name_too_long():
    with pytest.raises(ValidationError):
        TrainingConfig(**make_config(model_name="x" * 65))


def test_warmup_equal_to_epochs_ok():
    cfg = TrainingConfig(**make_config(epochs=10, warmup_epochs=10))
    assert cfg.warmup_epochs == 10


def test_warmup_greater_than_epochs_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig(**make_config(epochs=5, warmup_epochs=6))


# Nesting & serialization
def test_nested_layers_parsed_from_dicts():
    cfg = TrainingConfig(**make_config(layers=[{"units": 8}, {"units": 4}]))
    assert all(isinstance(layer, LayerConfig) for layer in cfg.layers)


def test_serialization_round_trip():
    cfg = TrainingConfig(**make_config(seed=7))
    dumped = cfg.model_dump()
    assert dumped["optimizer"] == "adam"
    assert dumped["layers"][0]["units"] == 16
    restored = TrainingConfig.model_validate(dumped)
    assert restored == cfg
