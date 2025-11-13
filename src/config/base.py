"""Configuration dataclasses and loaders."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingConfig:
    """Structured config for training runs."""
    name: str
    params: dict[str, Any]


def load_config(path: str) -> TrainingConfig:
    """Load a structured config object from disk."""
    ...
