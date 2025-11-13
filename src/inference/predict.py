"""Prediction-facing helpers (CLI/FastAPI can import these)."""

from typing import Any

def predict_image(image_path: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return species score, ID prediction, and open-set metadata for one image."""
    ...
