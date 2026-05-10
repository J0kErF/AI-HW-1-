import json
import os
from dataclasses import dataclass


@dataclass
class Config:
    frequencies: list[int]
    sample_rate: int
    duration: int
    window_size: int
    noise_level: float
    epochs: int
    learning_rate: float
    batch_size: int

def load_config(path: str = "config/setup.json") -> Config:
    """Load configuration from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path) as f:
        data = json.load(f)

    return Config(**data)
