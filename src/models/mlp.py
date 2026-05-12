from __future__ import annotations

from torch import nn


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int = 200, hidden_dim: int = 64, embedding_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        emb = self.backbone(x)
        return emb, self.head(emb)
