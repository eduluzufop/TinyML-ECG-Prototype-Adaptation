from __future__ import annotations

import numpy as np


def few_shot_feasible(y: np.ndarray, shots: int, required_classes: np.ndarray | None = None) -> tuple[bool, str]:
    labels = np.asarray(y)
    classes = np.asarray(required_classes if required_classes is not None else np.unique(labels))
    if labels.size == 0:
        return False, "sem amostras"
    for cls in classes:
        count = int(np.sum(labels == cls))
        if count < shots + 1:
            return False, f"classe {int(cls)} tem {count} amostras; minimo requerido={shots + 1}"
    return True, ""


def sample_few_shot(
    x: np.ndarray,
    y: np.ndarray,
    shots: int,
    seed: int = 42,
    required_classes: np.ndarray | None = None,
    support_classes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ok, reason = few_shot_feasible(y, shots, required_classes=required_classes)
    if not ok:
        raise ValueError(f"Few-shot inviavel para o conjunto alvo: {reason}")
    rng = np.random.default_rng(seed)
    support_idx = []
    classes_for_support = np.asarray(support_classes if support_classes is not None else np.unique(y))
    for c in classes_for_support:
        cls_idx = np.where(y == c)[0]
        k = min(shots, len(cls_idx))
        support_idx.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
    support_idx = np.array(sorted(set(support_idx)))
    query_idx = np.setdiff1d(np.arange(len(y)), support_idx)
    if len(query_idx) == 0:
        raise ValueError("Few-shot inviavel: conjunto query ficou vazio apos selecionar o suporte.")
    return x[support_idx], y[support_idx], x[query_idx], y[query_idx]
