#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.dataset.mitbih import infer_metadata_path, load_dataset_metadata, load_processed_npz
from src.dataset.splits import build_de_chazal_interpatient_split
from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs
from src.training.trainer import TrainConfig, evaluate_model, train_model
from src.utils.config import load_yaml
from src.utils.io import save_json
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    data = load_processed_npz(cfg["dataset_path"])
    metadata = load_dataset_metadata(cfg.get("metadata_path", infer_metadata_path(cfg["dataset_path"])))
    x, y = data["x"], data["y"]
    split = build_de_chazal_interpatient_split(
        data["records"],
        seed=int(cfg.get("seed", 42)),
        val_fraction=float(cfg.get("val_fraction", 0.1)),
    )
    x_train, y_train = x[split.train_idx], y[split.train_idx]
    x_val, y_val = x[split.val_idx], y[split.val_idx]
    x_test, y_test = x[split.test_idx], y[split.test_idx]

    model_kwargs = normalize_model_kwargs({"embedding_dim": int(cfg.get("embedding_dim", 32))})
    if cfg.get("model_config"):
        model_cfg = load_yaml(cfg["model_config"])
        model_kwargs = normalize_model_kwargs(model_kwargs, **model_cfg.get("model", {}))

    model = build_model_from_kwargs(num_classes=len(np.unique(y_train)), model_kwargs=model_kwargs)
    tcfg = TrainConfig(**cfg["training"])
    train_metrics = train_model(model, x_train, y_train, tcfg)
    val_metrics = evaluate_model(model, x_val, y_val) if len(x_val) else {}
    test_metrics = evaluate_model(model, x_test, y_test)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "baseline.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": int(len(np.unique(y_train))),
            "embedding_dim": int(model_kwargs["embedding_dim"]),
            "model_kwargs": model_kwargs,
            "split": split.to_metadata(),
            "dataset_metadata": metadata,
        },
        ckpt,
    )

    save_json(
        {
            "train": train_metrics,
            "val": val_metrics,
            "test_ds2": test_metrics,
            "split": split.to_metadata(),
        },
        cfg["results_path"],
    )
    print("Checkpoint salvo em", ckpt)


if __name__ == "__main__":
    main()
