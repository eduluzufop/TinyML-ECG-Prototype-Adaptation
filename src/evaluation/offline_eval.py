from __future__ import annotations

import copy
import time

import numpy as np
import torch

from src.personalization.last_layer_linear import adapt_linear_head
from src.personalization.prototypes import (
    build_prototypes,
    extract_embeddings,
    predict_with_prototypes,
    prototypes_from_embeddings,
)
from src.personalization.protocols import sample_few_shot
from src.training.metrics import compute_metrics


def evaluate_personalization(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, shots: int, seed: int = 42) -> dict:
    return evaluate_personalization_with_options(model, x, y, shots=shots, seed=seed)


def evaluate_personalization_with_options(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    shots: int,
    seed: int = 42,
    linear_kwargs: dict | None = None,
    prototype_kwargs: dict | None = None,
) -> dict:
    x_sup, y_sup, x_q, y_q = sample_few_shot(x, y, shots=shots, seed=seed)
    linear_kwargs = dict(linear_kwargs or {})
    prototype_kwargs = dict(prototype_kwargs or {})

    base_model = copy.deepcopy(model)
    base_model.eval()
    with torch.no_grad():
        _, logits = base_model(torch.from_numpy(x_q).float())
        pre_pred = logits.argmax(dim=1).cpu().numpy()
    pre = compute_metrics(y_q, pre_pred)

    linear_model = copy.deepcopy(model)
    lin_stats = adapt_linear_head(linear_model, x_sup, y_sup, **linear_kwargs)
    with torch.no_grad():
        _, logits = linear_model(torch.from_numpy(x_q).float())
        post_linear = compute_metrics(y_q, logits.argmax(dim=1).cpu().numpy())

    t0 = time.perf_counter()
    proto_model = copy.deepcopy(model)
    protos = build_prototypes(
        proto_model,
        x_sup,
        y_sup,
        num_classes=len(np.unique(y)),
    )
    if prototype_kwargs.get("normalize_embeddings", False):
        support_embeddings = extract_embeddings(proto_model, x_sup, normalize=True)
        protos = prototypes_from_embeddings(
            support_embeddings,
            y_sup,
            num_classes=len(np.unique(y)),
        )
    proto_pred = predict_with_prototypes(proto_model, x_q, protos, **prototype_kwargs)
    proto_ms = (time.perf_counter() - t0) * 1e3
    post_proto = compute_metrics(y_q, proto_pred)

    return {
        "shots": shots,
        "pre": pre,
        "post_linear": post_linear,
        "post_prototypes": post_proto,
        "linear_adaptation": lin_stats,
        "prototype_time_ms": proto_ms,
        "num_support_samples": int(len(y_sup)),
    }


def evaluate_restricted_normal_prototype(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    shots: int,
    global_prototypes: np.ndarray,
    seed: int = 42,
    normal_class: int = 0,
) -> dict:
    support_classes = np.asarray([normal_class], dtype=y.dtype)
    required_classes = support_classes
    x_sup, y_sup, x_q, y_q = sample_few_shot(
        x,
        y,
        shots=shots,
        seed=seed,
        required_classes=required_classes,
        support_classes=support_classes,
    )

    base_model = copy.deepcopy(model)
    base_model.eval()
    with torch.no_grad():
        _, logits = base_model(torch.from_numpy(x_q).float())
        pre_pred = logits.argmax(dim=1).cpu().numpy()
    pre = compute_metrics(y_q, pre_pred)

    t0 = time.perf_counter()
    proto_model = copy.deepcopy(model)
    support_embeddings = extract_embeddings(proto_model, x_sup)
    restricted_prototypes = prototypes_from_embeddings(
        support_embeddings,
        y_sup,
        num_classes=global_prototypes.shape[0],
        base_prototypes=global_prototypes,
    )
    proto_pred = predict_with_prototypes(proto_model, x_q, restricted_prototypes)
    proto_ms = (time.perf_counter() - t0) * 1e3
    post_proto = compute_metrics(y_q, proto_pred)

    return {
        "shots": shots,
        "pre": pre,
        "post_restricted_prototypes": post_proto,
        "restricted_prototype_time_ms": proto_ms,
        "num_support_samples": int(len(y_sup)),
        "support_class_distribution": {str(int(cls)): int((y_sup == cls).sum()) for cls in sorted(set(y_sup.tolist()))},
    }
