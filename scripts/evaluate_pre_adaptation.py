#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from src.dataset.mitbih import infer_metadata_path, load_dataset_metadata, load_processed_npz
from src.dataset.splits import build_de_chazal_interpatient_split
from src.evaluation.offline_eval import evaluate_personalization
from src.evaluation.report import build_gain_report
from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs
from src.personalization.protocols import few_shot_feasible
from src.utils.config import load_yaml
from src.utils.io import save_json


def mean_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    accumulator: dict[str, list[float]] = defaultdict(list)
    for item in items:
        for key, value in item.items():
            accumulator[key].append(float(value))
    return {key: float(sum(values) / len(values)) for key, values in accumulator.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    data = load_processed_npz(cfg["dataset_path"])
    metadata = load_dataset_metadata(cfg.get("metadata_path", infer_metadata_path(cfg["dataset_path"])))
    x, y = data["x"], data["y"]
    records = data["records"]
    split = build_de_chazal_interpatient_split(records, seed=int(cfg.get("seed", 42)))
    ds2_records = list(split.test_records)
    required_classes = torch.unique(torch.tensor(y)).cpu().numpy()

    ckpt = torch.load(cfg["checkpoint"], map_location="cpu")
    model_kwargs = normalize_model_kwargs(ckpt.get("model_kwargs", {"embedding_dim": int(ckpt.get("embedding_dim", 32))}))
    model = build_model_from_kwargs(num_classes=int(ckpt["num_classes"]), model_kwargs=model_kwargs)
    model.load_state_dict(ckpt["state_dict"])

    per_record = []
    skipped_records = []
    for record in ds2_records:
        mask = records == record
        x_target, y_target = x[mask], y[mask]
        ok, reason = few_shot_feasible(y_target, shots=int(cfg["shots"]), required_classes=required_classes)
        if not ok:
            skipped_records.append({"record": record, "reason": reason, "num_samples": int(len(y_target))})
            continue

        out = evaluate_personalization(
            model,
            x_target,
            y_target,
            shots=int(cfg["shots"]),
            seed=int(cfg.get("seed", 42)),
        )
        out["record"] = str(record)
        out["num_record_samples"] = int(len(y_target))
        out["class_distribution"] = {str(int(cls)): int((y_target == cls).sum()) for cls in sorted(set(y_target.tolist()))}
        out["linear_gain"] = build_gain_report(out["pre"], out["post_linear"])
        out["prototype_gain"] = build_gain_report(out["pre"], out["post_prototypes"])
        per_record.append(out)

    aggregate = {
        "pre": mean_metric_dict([item["pre"] for item in per_record]),
        "post_linear": mean_metric_dict([item["post_linear"] for item in per_record]),
        "post_prototypes": mean_metric_dict([item["post_prototypes"] for item in per_record]),
        "linear_gain": mean_metric_dict([item["linear_gain"] for item in per_record]),
        "prototype_gain": mean_metric_dict([item["prototype_gain"] for item in per_record]),
        "mean_linear_adaptation_time_ms": float(
            sum(item["linear_adaptation"]["adaptation_time_ms"] for item in per_record) / len(per_record)
        )
        if per_record
        else 0.0,
        "mean_prototype_time_ms": float(sum(item["prototype_time_ms"] for item in per_record) / len(per_record))
        if per_record
        else 0.0,
    }
    save_json(
        {
            "shots": int(cfg["shots"]),
            "protocol": metadata.get("split_protocol", "de_chazal_interpatient_ds1_ds2"),
            "target_records": ds2_records,
            "num_evaluated_records": len(per_record),
            "num_skipped_records": len(skipped_records),
            "skipped_records": skipped_records,
            "aggregate": aggregate,
            "per_record": per_record,
        },
        cfg["output_path"],
    )
    print("Avaliação salva em", cfg["output_path"])


if __name__ == "__main__":
    main()
