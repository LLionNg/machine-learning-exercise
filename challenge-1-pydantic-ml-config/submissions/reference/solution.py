"""Reference solution for Challenge 1: Pydantic ML training configuration."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class LayerConfig(BaseModel):
    units: int = Field(gt=0)
    activation: Literal["relu", "tanh", "sigmoid", "none"] = "relu"
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    model_name: str = Field(min_length=1, max_length=64)
    learning_rate: float = Field(gt=0.0, le=1.0)
    batch_size: int = Field(gt=0, le=4096)
    epochs: int = Field(gt=0, le=1000)
    warmup_epochs: int = Field(default=0, ge=0)
    optimizer: Literal["sgd", "adam", "adamw"] = "adam"
    layers: list[LayerConfig] = Field(min_length=1)
    device: Literal["cpu", "cuda"] = "cpu"
    seed: Optional[int] = None

    @field_validator("model_name", mode="before")
    @classmethod
    def _strip_name(cls, value: object) -> object:
        # Runs before length checks, so "  resnet  " -> "resnet" and an
        # all-whitespace name collapses to "" and is rejected by min_length.
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_warmup(self) -> "TrainingConfig":
        if self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs must not exceed epochs")
        return self
