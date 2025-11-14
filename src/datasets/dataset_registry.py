"""Dataset registry definitions and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from .animal_faces import AnimalFaceDataset
from .transforms import build_transforms


def load_dataset_metadata(registry_path: str) -> dict[str, Any]:
    """Load dataset metadata (splits, augmentations) from disk."""
    with Path(registry_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _instantiate_dataset(name: str, split: str, config: dict[str, Any]) -> Any:
    if name != "animal_faces":
        msg = f"Unknown dataset '{name}'. Only 'animal_faces' is scaffolded."
        raise ValueError(msg)

    transforms = build_transforms(stage=split, config=config)
    return AnimalFaceDataset(
        root=config["root"],
        split=split,
        transform=transforms,
    )


def build_dataloader(name: str, split: str, config: dict[str, Any]) -> DataLoader:
    """Return a dataloader for the requested dataset split."""
    dataset = _instantiate_dataset(name=name, split=split, config=config)
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=split == "train",
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
