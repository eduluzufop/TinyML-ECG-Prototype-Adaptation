from __future__ import annotations

import torch
from torch import nn


DEFAULT_MODEL_KWARGS = {
    "in_channels": 1,
    "conv_channels": (8, 16),
    "embedding_dim": 32,
}


def normalize_model_kwargs(model_kwargs: dict | None = None, **overrides: object) -> dict[str, object]:
    merged = dict(DEFAULT_MODEL_KWARGS)
    if model_kwargs:
        merged.update({key: value for key, value in model_kwargs.items() if key in DEFAULT_MODEL_KWARGS})
    merged.update({key: value for key, value in overrides.items() if value is not None and key in DEFAULT_MODEL_KWARGS})
    merged["in_channels"] = int(merged["in_channels"])
    merged["embedding_dim"] = int(merged["embedding_dim"])
    conv_channels = tuple(int(channel) for channel in merged["conv_channels"])
    if len(conv_channels) != 2:
        raise ValueError("conv_channels deve conter exatamente dois canais.")
    merged["conv_channels"] = conv_channels
    return merged


class TinyCNNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: tuple[int, int] = (8, 16),
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        c1, c2 = conv_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(c2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).squeeze(-1)
        return self.proj(h)


class BackboneWithHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 32,
        num_classes: int = 2,
        in_channels: int = 1,
        conv_channels: tuple[int, int] = (8, 16),
    ) -> None:
        super().__init__()
        self.backbone = TinyCNNBackbone(
            in_channels=in_channels,
            conv_channels=conv_channels,
            embedding_dim=embedding_dim,
        )
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(x)
        logits = self.head(emb)
        return emb, logits


def build_model_from_kwargs(num_classes: int, model_kwargs: dict | None = None) -> BackboneWithHead:
    normalized = normalize_model_kwargs(model_kwargs)
    return BackboneWithHead(num_classes=num_classes, **normalized)
