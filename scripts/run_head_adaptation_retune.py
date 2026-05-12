#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.dataset.mitbih import infer_metadata_path, load_dataset_metadata, load_processed_npz
from src.dataset.splits import build_de_chazal_interpatient_split
from src.evaluation.offline_eval import evaluate_personalization_with_options
from src.evaluation.report import build_gain_report
from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs
from src.personalization.protocols import few_shot_feasible
from src.utils.io import save_json


VARIANTS = [
    {
        "name": "medium",
        "label": "Medium",
        "checkpoint": "artifacts/checkpoints/architecture_sweep/medium.pt",
    },
    {
        "name": "large",
        "label": "Large",
        "checkpoint": "artifacts/checkpoints/architecture_sweep/large.pt",
    },
]

SGD_CONFIGS = [
    {"name": "sgd_lr0p01_s100", "lr": 0.01, "steps": 100},
    {"name": "sgd_lr0p02_s100", "lr": 0.02, "steps": 100},
    {"name": "sgd_lr0p05_s50", "lr": 0.05, "steps": 50},
    {"name": "sgd_lr0p05_s100", "lr": 0.05, "steps": 100},
    {"name": "sgd_lr0p10_s25", "lr": 0.10, "steps": 25},
    {"name": "sgd_lr0p10_s50", "lr": 0.10, "steps": 50},
]

PROTO_CONFIGS = [
    {"name": "proto_euclidean_raw", "distance": "euclidean", "normalize_embeddings": False},
    {"name": "proto_euclidean_norm", "distance": "euclidean", "normalize_embeddings": True},
    {"name": "proto_cosine_raw", "distance": "cosine", "normalize_embeddings": False},
    {"name": "proto_cosine_norm", "distance": "cosine", "normalize_embeddings": True},
]


def mean_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    accumulator: dict[str, list[float]] = defaultdict(list)
    for item in items:
        for key, value in item.items():
            accumulator[key].append(float(value))
    return {key: float(sum(values) / len(values)) for key, values in accumulator.items()}


def load_model(checkpoint_path: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = normalize_model_kwargs(
        ckpt.get("model_kwargs", {"embedding_dim": int(ckpt.get("embedding_dim", 32))})
    )
    model = build_model_from_kwargs(num_classes=int(ckpt["num_classes"]), model_kwargs=model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def evaluate_variant_config(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    records: np.ndarray,
    ds2_records: list[str],
    shots: int,
    seed: int,
    linear_kwargs: dict | None = None,
    prototype_kwargs: dict | None = None,
) -> dict[str, object]:
    required_classes = torch.unique(torch.tensor(y)).cpu().numpy()
    per_record = []
    skipped_records = []
    for record in ds2_records:
        mask = records == record
        x_target, y_target = x[mask], y[mask]
        ok, reason = few_shot_feasible(y_target, shots=shots, required_classes=required_classes)
        if not ok:
            skipped_records.append({"record": record, "reason": reason, "num_samples": int(len(y_target))})
            continue

        out = evaluate_personalization_with_options(
            model,
            x_target,
            y_target,
            shots=shots,
            seed=seed,
            linear_kwargs=linear_kwargs,
            prototype_kwargs=prototype_kwargs,
        )
        out["record"] = str(record)
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
    return {
        "shots": shots,
        "num_evaluated_records": len(per_record),
        "num_skipped_records": len(skipped_records),
        "skipped_records": skipped_records,
        "aggregate": aggregate,
    }


def best_by_key(rows: list[dict[str, object]], metric_key: str) -> dict[str, object]:
    return max(rows, key=lambda row: float(row[metric_key]))


def build_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Head Adaptation Retuning",
        "",
        "| Variant | Shot | Best SGD | SGD F1 | Best Proto | Proto F1 | Proto > SGD? |",
        "|---|---:|---|---:|---|---:|:---:|",
    ]
    for variant in summary["variants"]:
        for shot_key, shot_data in variant["shots"].items():
            best_sgd = shot_data["best_sgd"]
            best_proto = shot_data["best_proto"]
            lines.append(
                "| {label} | {shot} | {sgd_name} | {sgd_f1:.3f} | {proto_name} | {proto_f1:.3f} | {flag} |".format(
                    label=variant["label"],
                    shot=shot_key.replace("shot_", ""),
                    sgd_name=best_sgd["name"],
                    sgd_f1=best_sgd["f1_macro"],
                    proto_name=best_proto["name"],
                    proto_f1=best_proto["f1_macro"],
                    flag="yes" if best_proto["f1_macro"] > best_sgd["f1_macro"] else "no",
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default="data/processed/mitbih_binary.npz")
    ap.add_argument("--metadata-path", default=None)
    ap.add_argument("--output-dir", default="results/head_retune")
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    metadata_path = args.metadata_path or infer_metadata_path(args.dataset_path)
    data = load_processed_npz(args.dataset_path)
    metadata = load_dataset_metadata(metadata_path)
    x, y, records = data["x"], data["y"], data["records"]
    split = build_de_chazal_interpatient_split(records, seed=args.seed)
    ds2_records = list(split.test_records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants_summary = []
    for variant in VARIANTS:
        model = load_model(variant["checkpoint"])
        variant_result = {"name": variant["name"], "label": variant["label"], "checkpoint": variant["checkpoint"], "shots": {}}

        for shot in args.shots:
            sgd_rows = []
            for cfg in SGD_CONFIGS:
                result = evaluate_variant_config(
                    model,
                    x,
                    y,
                    records,
                    ds2_records,
                    shots=int(shot),
                    seed=args.seed,
                    linear_kwargs={"lr": cfg["lr"], "steps": cfg["steps"]},
                    prototype_kwargs={"distance": "euclidean", "normalize_embeddings": False},
                )
                sgd_rows.append(
                    {
                        "name": cfg["name"],
                        "linear_kwargs": {"lr": cfg["lr"], "steps": cfg["steps"]},
                        "f1_macro": float(result["aggregate"]["post_linear"]["f1_macro"]),
                        "accuracy": float(result["aggregate"]["post_linear"]["accuracy"]),
                        "pre_f1_macro": float(result["aggregate"]["pre"]["f1_macro"]),
                        "mean_adaptation_time_ms": float(result["aggregate"]["mean_linear_adaptation_time_ms"]),
                    }
                )

            proto_rows = []
            for cfg in PROTO_CONFIGS:
                result = evaluate_variant_config(
                    model,
                    x,
                    y,
                    records,
                    ds2_records,
                    shots=int(shot),
                    seed=args.seed,
                    linear_kwargs={"lr": 0.05, "steps": 50},
                    prototype_kwargs={
                        "distance": cfg["distance"],
                        "normalize_embeddings": cfg["normalize_embeddings"],
                    },
                )
                proto_rows.append(
                    {
                        "name": cfg["name"],
                        "prototype_kwargs": {
                            "distance": cfg["distance"],
                            "normalize_embeddings": cfg["normalize_embeddings"],
                        },
                        "f1_macro": float(result["aggregate"]["post_prototypes"]["f1_macro"]),
                        "accuracy": float(result["aggregate"]["post_prototypes"]["accuracy"]),
                        "pre_f1_macro": float(result["aggregate"]["pre"]["f1_macro"]),
                        "mean_adaptation_time_ms": float(result["aggregate"]["mean_prototype_time_ms"]),
                    }
                )

            shot_key = f"shot_{shot}"
            variant_result["shots"][shot_key] = {
                "baseline_pre_f1_macro": float(sgd_rows[0]["pre_f1_macro"]),
                "sgd_grid": sgd_rows,
                "prototype_grid": proto_rows,
                "best_sgd": best_by_key(sgd_rows, "f1_macro"),
                "best_proto": best_by_key(proto_rows, "f1_macro"),
            }
        save_json(variant_result, output_dir / f"{variant['name']}_retune.json")
        variants_summary.append(variant_result)

    summary = {
        "protocol": metadata.get("split_protocol", "de_chazal_interpatient_ds1_ds2"),
        "shots": [int(shot) for shot in args.shots],
        "variants": variants_summary,
    }
    save_json(summary, output_dir / "summary.json")
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    print("Retuning salvo em", output_dir)


if __name__ == "__main__":
    main()
