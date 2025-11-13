"""Closed-set training entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

from src.config.base import TrainingConfig
from src.datasets.dataset_registry import build_dataloader
from src.models.backbones import build_backbone


def _prepare_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def train_closed_set(config: dict[str, Any] | TrainingConfig) -> None:
    """Run the closed-set baseline training loop."""
    if isinstance(config, TrainingConfig):
        config = config.as_dict()
    data_cfg = config["data"]
    model_cfg = config["model"]
    trainer_cfg = config["trainer"]

    train_loader = build_dataloader(
        name=data_cfg.get("dataset_name", "gorilla_faces"),
        split="train",
        config=data_cfg,
    )
    val_loader = build_dataloader(
        name=data_cfg.get("dataset_name", "gorilla_faces"),
        split="val",
        config=data_cfg,
    )

    model = build_backbone(
        model_name=model_cfg.get("backbone", "resnet18"),
        embedding_dim=model_cfg.get("embedding_dim", 256),
        pretrained=model_cfg.get("pretrained", True),
    )
    classifier = nn.Linear(model_cfg.get("embedding_dim", 256), data_cfg.get("num_classes", 1))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=trainer_cfg.get("lr", 3e-4),
    )

    device = _prepare_device(trainer_cfg.get("device", "cpu"))
    model.to(device)
    classifier.to(device)

    max_epochs = trainer_cfg.get("max_epochs", 1)
    for epoch in range(max_epochs):
        model.train()
        classifier.train()
        for batch in train_loader:
            # Placeholder for forward/backward pass.
            _ = batch
            # TODO: implement actual training logic (forward pass, loss, backward, step).
            pass

        model.eval()
        classifier.eval()
        with torch.no_grad():
            for batch in val_loader:
                _ = batch
                # TODO: implement validation logic.
                pass

    checkpoint_dir = Path(trainer_cfg.get("checkpoint_dir", "artifacts"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "closed_set_skeleton.pt"
    torch.save({"model_state": model.state_dict(), "classifier_state": classifier.state_dict()}, checkpoint_path)
