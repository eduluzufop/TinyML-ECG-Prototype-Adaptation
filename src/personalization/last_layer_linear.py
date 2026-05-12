from __future__ import annotations

import time

import numpy as np
import torch


def adapt_linear_head(
    model: torch.nn.Module,
    x_support: np.ndarray,
    y_support: np.ndarray,
    steps: int = 50,
    lr: float = 0.05,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    opt = torch.optim.SGD(
        model.head.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    xs = torch.from_numpy(x_support).float()
    ys = torch.from_numpy(y_support).long()
    model.train()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        _, logits = model(xs)
        loss = criterion(logits, ys)
        loss.backward()
        opt.step()
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return {
        "adaptation_time_ms": elapsed_ms,
        "updates": steps,
        "lr": float(lr),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
    }
