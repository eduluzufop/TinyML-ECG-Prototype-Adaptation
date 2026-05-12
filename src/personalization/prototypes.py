from __future__ import annotations

import numpy as np
import torch


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, a_min=eps, a_max=None)


def extract_embeddings(model: torch.nn.Module, x: np.ndarray, normalize: bool = False) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        emb, _ = model(torch.from_numpy(x).float())
    emb_np = emb.cpu().numpy()
    if normalize:
        emb_np = _l2_normalize_np(emb_np)
    return emb_np


def prototypes_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    base_prototypes: np.ndarray | None = None,
) -> np.ndarray:
    protos = (
        np.array(base_prototypes, dtype=np.float32, copy=True)
        if base_prototypes is not None
        else np.zeros((num_classes, embeddings.shape[1]), dtype=np.float32)
    )
    for c in range(num_classes):
        cls = embeddings[labels == c]
        if len(cls) > 0:
            protos[c] = cls.mean(axis=0)
    return protos


def build_prototypes(model: torch.nn.Module, x_support: np.ndarray, y_support: np.ndarray, num_classes: int) -> np.ndarray:
    emb_np = extract_embeddings(model, x_support)
    return prototypes_from_embeddings(emb_np, y_support, num_classes=num_classes)


def predict_with_prototypes(
    model: torch.nn.Module,
    x_query: np.ndarray,
    prototypes: np.ndarray,
    distance: str = "euclidean",
    normalize_embeddings: bool = False,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        emb, _ = model(torch.from_numpy(x_query).float())
    emb_t = emb.float()
    proto_t = torch.from_numpy(prototypes).float()
    if normalize_embeddings:
        emb_t = torch.nn.functional.normalize(emb_t, p=2, dim=1)
        proto_t = torch.nn.functional.normalize(proto_t, p=2, dim=1)
    if distance == "euclidean":
        scores = -torch.cdist(emb_t, proto_t)
    elif distance == "cosine":
        scores = emb_t @ proto_t.T
    else:
        raise ValueError(f"Distancia de prototipos nao suportada: {distance}")
    return torch.argmax(scores, dim=1).cpu().numpy()
