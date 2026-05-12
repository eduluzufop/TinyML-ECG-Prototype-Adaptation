from __future__ import annotations

import torch


def prototype_predict(embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    dists = torch.cdist(embeddings, prototypes)
    return torch.argmin(dists, dim=1)
