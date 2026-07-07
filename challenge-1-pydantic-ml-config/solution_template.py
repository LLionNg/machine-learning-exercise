from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class LayerConfig(BaseModel):
    # TODO: must be > 0
    units: int  

    # TODO: Literal["relu", "tanh", "sigmoid", "none"]
    activation: str = "relu"  

    # TODO: constrain to [0.0, 1.0]
    dropout: float = 0.0  


class TrainingConfig(BaseModel):
    # TODO: non-empty, <= 64 chars (strip whitespace, enforced with field_validator)
    model_name: str  

    # TODO: 0.0 < lr <= 1.0
    learning_rate: float  

    # TODO: 0 < batch_size <= 4096
    batch_size: int  

    # TODO: 0 < epochs <= 1000
    epochs: int  

    # TODO: must be >= 0 (and <= epochs, enforced below with model_validator)
    warmup_epochs: int = 0  

    # TODO: Literal["sgd", "adam", "adamw"]
    optimizer: str = "adam"  

    # TODO: at least one
    layers: list[LayerConfig]  
    
    # TODO: Literal["cpu", "cuda"]
    device: str = "cpu"  
    seed: Optional[int] = None

    # TODO: field_validator(mode="before") to strip model_name
    # TODO: model_validator(mode="after") to enforce warmup_epochs <= epochs
