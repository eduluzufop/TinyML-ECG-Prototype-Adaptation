#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs
from src.personalization.last_layer_linear import adapt_linear_head
from src.personalization.prototypes import build_prototypes, predict_with_prototypes
from src.training.metrics import compute_metrics
from src.utils.io import save_json


def mean_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    acc: dict[str, list[float]] = defaultdict(list)
    for item in items:
        for key, value in item.items():
            acc[key].append(float(value))
    return {key: float(sum(values) / len(values)) for key, values in acc.items()}


def load_npz_subset(path: str | Path, indices: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.uint8)
    if indices is not None:
        x = x[indices]
        y = y[indices]
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/firmware/ondevice_batch_1shot.json")
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/baseline.pt")
    ap.add_argument("--out", default="results/firmware/ondevice_batch_1shot_reference.json")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_kwargs = normalize_model_kwargs(ckpt.get("model_kwargs", {"embedding_dim": int(ckpt.get("embedding_dim", 32))}))
    model = build_model_from_kwargs(num_classes=int(ckpt["num_classes"]), model_kwargs=model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    per_episode = []
    for episode in summary["episodes"]:
        x_sup, y_sup = load_npz_subset(episode["support_path"])
        x_query, y_query = load_npz_subset(episode["query_path"], indices=episode["selected_query_indices"])

        pre_model = copy.deepcopy(model)
        with torch.no_grad():
            _, logits = pre_model(torch.from_numpy(x_query).float())
            pre_pred = logits.argmax(dim=1).cpu().numpy()
        pre = compute_metrics(y_query, pre_pred)

        sgd_model = copy.deepcopy(model)
        linear_stats = adapt_linear_head(sgd_model, x_sup, y_sup)
        with torch.no_grad():
            _, logits = sgd_model(torch.from_numpy(x_query).float())
            sgd_pred = logits.argmax(dim=1).cpu().numpy()
        post_linear = compute_metrics(y_query, sgd_pred)

        proto_model = copy.deepcopy(model)
        prototypes = build_prototypes(proto_model, x_sup, y_sup, num_classes=int(ckpt["num_classes"]))
        proto_pred = predict_with_prototypes(proto_model, x_query, prototypes)
        post_proto = compute_metrics(y_query, proto_pred)

        per_episode.append(
            {
                "record": episode["record"],
                "query_count": int(len(y_query)),
                "pre": pre,
                "post_linear": post_linear,
                "post_prototypes": post_proto,
                "linear_adaptation": linear_stats,
            }
        )

    aggregate = {
        "pre": mean_metric_dict([item["pre"] for item in per_episode]),
        "post_linear": mean_metric_dict([item["post_linear"] for item in per_episode]),
        "post_prototypes": mean_metric_dict([item["post_prototypes"] for item in per_episode]),
        "mean_linear_adaptation_time_ms": float(
            sum(item["linear_adaptation"]["adaptation_time_ms"] for item in per_episode) / len(per_episode)
        )
        if per_episode
        else 0.0,
    }
    save_json(
        {
            "summary_path": str(Path(args.summary)),
            "checkpoint": str(Path(args.checkpoint)),
            "num_episodes": len(per_episode),
            "aggregate": aggregate,
            "per_episode": per_episode,
        },
        args.out,
    )
    print("Referência host salva em", args.out)


if __name__ == "__main__":
    main()
