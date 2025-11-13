"""Backbone network definitions for gorilla face embeddings."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models


class ResNetEmbedding(nn.Module):
    """Thin wrapper that exposes a ResNet backbone with a projection head."""

    def __init__(self, embedding_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features
        self.projection = nn.Linear(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return L2-normalized embeddings."""
        feats = self.feature_extractor(x)
        feats = feats.flatten(1)
        embeddings = self.projection(feats)
        return nn.functional.normalize(embeddings, dim=-1)


def build_backbone(model_name: str, embedding_dim: int, pretrained: bool = True) -> Any:
    """Return a backbone model configured for the requested embedding size."""
    if model_name.lower() != "resnet18":
        msg = f"Unsupported backbone '{model_name}'. Only 'resnet18' is scaffolded."
        raise ValueError(msg)
    return ResNetEmbedding(embedding_dim=embedding_dim, pretrained=pretrained)
