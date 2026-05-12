#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from src.dataset.mitbih import infer_metadata_path, load_dataset_metadata, load_processed_npz
from src.dataset.splits import build_de_chazal_interpatient_split
from src.evaluation.offline_eval import evaluate_restricted_normal_prototype
from src.evaluation.report import build_gain_report
from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs
from src.personalization.protocols import few_shot_feasible
from src.personalization.prototypes import build_prototypes
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
    ap.add_argument("--dataset-path", default="data/processed/mitbih_binary.npz")
    ap.add_argument("--metadata-path", default=None)
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/baseline.pt")
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--normal-class", type=int, default=0)
    args = ap.parse_args()

    metadata_path = args.metadata_path or infer_metadata_path(args.dataset_path)
    data = load_processed_npz(args.dataset_path)
    metadata = load_dataset_metadata(metadata_path)
    x, y = data["x"], data["y"]
    records = data["records"]

    split = build_de_chazal_interpatient_split(records, seed=args.seed)
    ds1_train_mask = np.isin(records, split.train_records)
    ds2_records = list(split.test_records)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_kwargs = normalize_model_kwargs(ckpt.get("model_kwargs", {"embedding_dim": int(ckpt.get("embedding_dim", 32))}))
    model = build_model_from_kwargs(num_classes=int(ckpt["num_classes"]), model_kwargs=model_kwargs)
    model.load_state_dict(ckpt["state_dict"])

    global_prototypes = build_prototypes(
        model,
        x[ds1_train_mask],
        y[ds1_train_mask],
        num_classes=int(ckpt["num_classes"]),
    )

    per_record = []
    skipped_records = []
    required_classes = np.asarray([args.normal_class], dtype=y.dtype)

    for record in ds2_records:
        mask = records == record
        x_target, y_target = x[mask], y[mask]
        ok, reason = few_shot_feasible(y_target, shots=args.shots, required_classes=required_classes)
        if not ok:
            skipped_records.append({"record": record, "reason": reason, "num_samples": int(len(y_target))})
            continue

        out = evaluate_restricted_normal_prototype(
            model,
            x_target,
            y_target,
            shots=args.shots,
            global_prototypes=global_prototypes,
            seed=args.seed,
            normal_class=args.normal_class,
        )
        out["record"] = str(record)
        out["num_record_samples"] = int(len(y_target))
        out["class_distribution"] = {str(int(cls)): int((y_target == cls).sum()) for cls in sorted(set(y_target.tolist()))}
        out["restricted_prototype_gain"] = build_gain_report(out["pre"], out["post_restricted_prototypes"])
        per_record.append(out)

    aggregate = {
        "pre": mean_metric_dict([item["pre"] for item in per_record]),
        "post_restricted_prototypes": mean_metric_dict([item["post_restricted_prototypes"] for item in per_record]),
        "restricted_prototype_gain": mean_metric_dict([item["restricted_prototype_gain"] for item in per_record]),
        "mean_restricted_prototype_time_ms": float(
            sum(item["restricted_prototype_time_ms"] for item in per_record) / len(per_record)
        )
        if per_record
        else 0.0,
    }

    save_json(
        {
            "shots": int(args.shots),
            "protocol": metadata.get("split_protocol", "de_chazal_interpatient_ds1_ds2"),
            "scenario": "target_normal_only_support__global_arrhythmia_prototype",
            "target_records": ds2_records,
            "num_evaluated_records": len(per_record),
            "num_skipped_records": len(skipped_records),
            "skipped_records": skipped_records,
            "global_prototype_source": {
                "records": list(split.train_records),
                "normal_class": int(args.normal_class),
            },
            "aggregate": aggregate,
            "per_record": per_record,
        },
        args.output_path,
    )
    print("Avaliação restrita salva em", args.output_path)


if __name__ == "__main__":
    main()
