#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.dataset.mitbih import infer_metadata_path, load_dataset_metadata, load_processed_npz
from src.dataset.splits import build_de_chazal_interpatient_split
from src.evaluation.offline_eval import evaluate_personalization
from src.evaluation.report import build_gain_report
from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs
from src.personalization.protocols import few_shot_feasible
from src.training.trainer import TrainConfig, evaluate_model, train_model
from src.utils.config import load_yaml
from src.utils.io import save_json
from src.utils.seed import set_seed


VARIANTS = [
    {
        "name": "tiny",
        "label": "Base",
        "model_config": "configs/model/cnn1d_tiny.yaml",
        "expected_scale": "1.0x",
    },
    {
        "name": "medium",
        "label": "Medium",
        "model_config": "configs/model/cnn1d_medium.yaml",
        "expected_scale": "~2.0x",
    },
    {
        "name": "large",
        "label": "Large",
        "model_config": "configs/model/cnn1d_large.yaml",
        "expected_scale": "~4.0x",
    },
]


def mean_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    accumulator: dict[str, list[float]] = defaultdict(list)
    for item in items:
        for key, value in item.items():
            accumulator[key].append(float(value))
    return {key: float(sum(values) / len(values)) for key, values in accumulator.items()}


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def estimate_macs_per_beat(model_kwargs: dict[str, object], num_classes: int) -> int:
    signal_length = 200
    kernel_size = 5
    c1, c2 = (int(channel) for channel in model_kwargs["conv_channels"])
    embedding_dim = int(model_kwargs["embedding_dim"])
    conv1 = signal_length * c1 * int(model_kwargs["in_channels"]) * kernel_size
    post_pool_1 = signal_length // 2
    conv2 = post_pool_1 * c2 * c1 * kernel_size
    proj = c2 * embedding_dim
    head = embedding_dim * num_classes
    return int(conv1 + conv2 + proj + head)


def evaluate_ds2_protocol(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    records: np.ndarray,
    ds2_records: list[str],
    shots: int,
    seed: int,
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

        out = evaluate_personalization(model, x_target, y_target, shots=shots, seed=seed)
        out["record"] = str(record)
        out["num_record_samples"] = int(len(y_target))
        out["class_distribution"] = {
            str(int(cls)): int((y_target == cls).sum()) for cls in sorted(set(y_target.tolist()))
        }
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
        "per_record": per_record,
    }


def build_markdown_report(summary: dict[str, object]) -> str:
    lines = [
        "# Architecture Sweep",
        "",
        "## Variants",
        "",
        "| Variant | Scale | Params | MACs/beat | DS2 test acc | DS2 test F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in summary["variants"]:
        lines.append(
            "| {label} | {scale} | {params} | {macs} | {acc:.3f} | {f1:.3f} |".format(
                label=variant["label"],
                scale=variant["parameter_scale_vs_tiny"],
                params=variant["parameter_count"],
                macs=variant["macs_per_beat"],
                acc=variant["test_ds2"]["accuracy"],
                f1=variant["test_ds2"]["f1_macro"],
            )
        )

    lines.extend(
        [
            "",
            "## Few-Shot Comparison",
            "",
            "| Variant | Shot | Pre F1 | Linear SGD F1 | Proto F1 | Proto > SGD? |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for variant in summary["variants"]:
        for shot_name, shot_data in variant["few_shot"].items():
            agg = shot_data["aggregate"]
            proto_better = agg["post_prototypes"]["f1_macro"] > agg["post_linear"]["f1_macro"]
            lines.append(
                "| {label} | {shot} | {pre:.3f} | {sgd:.3f} | {proto:.3f} | {flag} |".format(
                    label=variant["label"],
                    shot=shot_name.replace("shot_", ""),
                    pre=agg["pre"]["f1_macro"],
                    sgd=agg["post_linear"]["f1_macro"],
                    proto=agg["post_prototypes"]["f1_macro"],
                    flag="yes" if proto_better else "no",
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-config", default="configs/training/baseline.yaml")
    ap.add_argument("--output-dir", default="results/architecture_sweep")
    ap.add_argument("--checkpoint-dir", default="artifacts/checkpoints/architecture_sweep")
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    args = ap.parse_args()

    train_cfg = load_yaml(args.training_config)
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    dataset_path = train_cfg["dataset_path"]
    metadata_path = train_cfg.get("metadata_path", infer_metadata_path(dataset_path))
    data = load_processed_npz(dataset_path)
    metadata = load_dataset_metadata(metadata_path)
    x, y, records = data["x"], data["y"], data["records"]
    split = build_de_chazal_interpatient_split(
        records,
        seed=seed,
        val_fraction=float(train_cfg.get("val_fraction", 0.1)),
    )
    x_train, y_train = x[split.train_idx], y[split.train_idx]
    x_val, y_val = x[split.val_idx], y[split.val_idx]
    x_test, y_test = x[split.test_idx], y[split.test_idx]
    ds2_records = list(split.test_records)

    tcfg = TrainConfig(**train_cfg["training"])
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    variants_summary = []
    baseline_params = None
    for variant in VARIANTS:
        set_seed(seed)
        model_cfg = load_yaml(variant["model_config"])
        model_kwargs = normalize_model_kwargs(model_cfg.get("model", {}))
        model = build_model_from_kwargs(num_classes=len(np.unique(y_train)), model_kwargs=model_kwargs)

        train_metrics = train_model(model, x_train, y_train, tcfg)
        val_metrics = evaluate_model(model, x_val, y_val) if len(x_val) else {}
        test_metrics = evaluate_model(model, x_test, y_test)

        parameter_count = count_parameters(model)
        if baseline_params is None:
            baseline_params = parameter_count
        parameter_scale = float(parameter_count / baseline_params)
        macs_per_beat = estimate_macs_per_beat(model_kwargs, int(len(np.unique(y_train))))

        ckpt_path = checkpoint_dir / f"{variant['name']}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_classes": int(len(np.unique(y_train))),
                "embedding_dim": int(model_kwargs["embedding_dim"]),
                "model_kwargs": model_kwargs,
                "split": split.to_metadata(),
                "dataset_metadata": metadata,
            },
            ckpt_path,
        )

        few_shot_results = {}
        for shot in args.shots:
            result = evaluate_ds2_protocol(
                model,
                x,
                y,
                records,
                ds2_records,
                shots=int(shot),
                seed=seed,
            )
            shot_key = f"shot_{shot}"
            few_shot_results[shot_key] = result
            save_json(result, output_dir / f"{variant['name']}_{shot_key}.json")

        variant_summary = {
            "name": variant["name"],
            "label": variant["label"],
            "model_config": variant["model_config"],
            "expected_scale": variant["expected_scale"],
            "model_kwargs": model_kwargs,
            "parameter_count": parameter_count,
            "parameter_scale_vs_tiny": f"{parameter_scale:.2f}x",
            "macs_per_beat": macs_per_beat,
            "train": train_metrics,
            "val": val_metrics,
            "test_ds2": test_metrics,
            "checkpoint": str(ckpt_path),
            "few_shot": few_shot_results,
        }
        save_json(variant_summary, output_dir / f"{variant['name']}_summary.json")
        variants_summary.append(variant_summary)

    summary = {
        "protocol": metadata.get("split_protocol", "de_chazal_interpatient_ds1_ds2"),
        "training_config": args.training_config,
        "shots": [int(shot) for shot in args.shots],
        "variants": variants_summary,
    }
    save_json(summary, output_dir / "summary.json")
    (output_dir / "summary.md").write_text(build_markdown_report(summary), encoding="utf-8")
    print("Sweep salvo em", output_dir)


if __name__ == "__main__":
    main()
