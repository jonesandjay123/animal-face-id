"""Dataset registry definitions and helpers."""

from typing import Any

def load_dataset_metadata(registry_path: str) -> dict[str, Any]:
    """Load dataset metadata (splits, augmentations) from disk."""
    ...


def build_dataloader(name: str, split: str, config: dict[str, Any]) -> Any:
    """Return a dataloader for the requested dataset split."""
    ...
