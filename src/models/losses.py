"""Metric-learning loss builders."""

from typing import Any

def build_loss(name: str, margin: float | None = None, scale: float | None = None) -> Any:
    """Return a configured loss object for training."""
    ...
