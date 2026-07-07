from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class LayerConfig(BaseModel):
    # TODO: must be > 0
    units: int = Field(gt=0)

    # TODO: Literal["relu", "tanh", "sigmoid", "none"]
    activation: Literal["relu", "tanh", "sigmoid", "none"] = "relu"  

    # TODO: constrain to [0.0, 1.0]
    dropout: float = Field(default=0.0, ge=0.0, le=1.0) 


class TrainingConfig(BaseModel):
    # TODO: non-empty, <= 64 chars (strip whitespace, enforced with field_validator)
    model_name: str = Field(min_length=1, max_length=64)

    # TODO: 0.0 < lr <= 1.0
    learning_rate: float = Field(gt=0.0, le=1.0)

    # TODO: 0 < batch_size <= 4096
    batch_size: int = Field(gt=0, le=4096)

    # TODO: 0 < epochs <= 1000
    epochs: int = Field(gt=0, le=1000)

    # TODO: must be >= 0 (and <= epochs, enforced below with model_validator)
    warmup_epochs: int = Field(default=0, ge=0)

    # TODO: Literal["sgd", "adam", "adamw"]
    optimizer: Literal["sgd", "adam", "adamw"] = "adam"  

    # TODO: at least one
    layers: list[LayerConfig] = Field(min_length=1)
    
    # TODO: Literal["cpu", "cuda"]
    device: Literal["cpu", "cuda"] = "cpu"  
    seed: Optional[int] = None

    # TODO: field_validator(mode="before") to strip model_name
    # TODO: model_validator(mode="after") to enforce warmup_epochs <= epochs
    @field_validator("model_name", mode="before")
    @classmethod
    def _strip_name(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_warmup(self) -> "TrainingConfig":
        if self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs must not exceed epochs")
        return self

