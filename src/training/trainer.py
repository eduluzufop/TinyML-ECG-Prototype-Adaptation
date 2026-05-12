from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.losses import get_classification_loss
from src.training.metrics import compute_metrics


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0


def train_model(model: torch.nn.Module, x_train: np.ndarray, y_train: np.ndarray, cfg: TrainConfig) -> dict[str, float]:
    ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = get_classification_loss()
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            opt.zero_grad()
            _, logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
    return evaluate_model(model, x_train, y_train)


def evaluate_model(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        _, logits = model(torch.from_numpy(x).float())
        pred = logits.argmax(dim=1).cpu().numpy()
    return compute_metrics(y, pred)
