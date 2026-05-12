#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.export.export_headers import export_named_arrays_header
from src.utils.io import save_json


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.uint8)


def select_query_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    total = int(labels.shape[0])
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)

    classes, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)

    selected_counts = np.zeros_like(counts, dtype=np.int64)
    if max_samples >= len(classes):
        selected_counts[:] = 1
    remaining_budget = max_samples - int(selected_counts.sum())
    remaining_capacity = counts.astype(np.int64) - selected_counts

    if remaining_budget > 0:
        cap_sum = int(remaining_capacity.sum())
        if cap_sum > 0:
            fractional_target = remaining_capacity / cap_sum * remaining_budget
            extra = np.floor(fractional_target).astype(np.int64)
            extra = np.minimum(extra, remaining_capacity)
            selected_counts += extra
            remaining_budget -= int(extra.sum())
            remaining_capacity = counts.astype(np.int64) - selected_counts

        while remaining_budget > 0:
            candidates = np.where(remaining_capacity > 0)[0]
            if candidates.size == 0:
                break
            priority = []
            for idx in candidates.tolist():
                frac = 0.0
                if remaining_capacity.sum() > 0:
                    frac = float(remaining_capacity[idx]) / float(remaining_capacity.sum())
                priority.append((-frac, int(classes[idx]), idx))
            priority.sort()
            pick = priority[0][2]
            selected_counts[pick] += 1
            remaining_capacity[pick] -= 1
            remaining_budget -= 1

    picked: list[int] = []
    for cls, count in zip(classes.tolist(), selected_counts.tolist()):
        cls_idx = np.where(labels == cls)[0]
        if count >= len(cls_idx):
            chosen = cls_idx
        elif count > 0:
            chosen = np.sort(rng.choice(cls_idx, size=count, replace=False))
        else:
            chosen = np.array([], dtype=np.int64)
        picked.extend(int(i) for i in chosen.tolist())
    return np.array(sorted(picked), dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="firmware/psoc6/generated/replay_support_1shot/manifest.json")
    ap.add_argument("--out", default="firmware/psoc6/generated/benchmark_replay.h")
    ap.add_argument("--summary-out", default="results/firmware/ondevice_batch_1shot.json")
    ap.add_argument("--max-query-samples-per-record", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    support_chunks: list[np.ndarray] = []
    support_label_chunks: list[np.ndarray] = []
    query_chunks: list[np.ndarray] = []
    query_label_chunks: list[np.ndarray] = []
    support_offsets: list[int] = []
    support_counts: list[int] = []
    query_offsets: list[int] = []
    query_counts: list[int] = []
    record_ids: list[int] = []
    episodes: list[dict[str, object]] = []

    support_cursor = 0
    query_cursor = 0
    signal_length = None
    max_support = 0
    max_query = 0

    for item in manifest["exported_records"]:
        x_support, y_support = load_npz(item["support_path"])
        x_query, y_query = load_npz(item["query_path"])
        signal_length = signal_length or int(x_support.shape[-1])
        if int(x_support.shape[-1]) != signal_length or int(x_query.shape[-1]) != signal_length:
            raise ValueError(f"Comprimento inconsistente para registro {item['record']}")

        record_seed = int(args.seed) + int(item["record"])
        selected_query_idx = select_query_indices(
            y_query,
            max_samples=int(args.max_query_samples_per_record),
            seed=record_seed,
        )
        x_query_sel = x_query[selected_query_idx]
        y_query_sel = y_query[selected_query_idx]

        support_flat = x_support[:, 0, :]
        query_flat = x_query_sel[:, 0, :]
        support_chunks.append(support_flat)
        support_label_chunks.append(y_support)
        query_chunks.append(query_flat)
        query_label_chunks.append(y_query_sel)

        support_offsets.append(support_cursor)
        support_counts.append(int(y_support.shape[0]))
        query_offsets.append(query_cursor)
        query_counts.append(int(y_query_sel.shape[0]))
        record_ids.append(int(item["record"]))

        max_support = max(max_support, int(y_support.shape[0]))
        max_query = max(max_query, int(y_query_sel.shape[0]))

        support_cursor += int(y_support.shape[0])
        query_cursor += int(y_query_sel.shape[0])

        episodes.append(
            {
                "record": str(item["record"]),
                "support_path": item["support_path"],
                "query_path": item["query_path"],
                "support_offset": support_offsets[-1],
                "support_count": support_counts[-1],
                "query_offset": query_offsets[-1],
                "query_count_selected": query_counts[-1],
                "query_count_total": int(y_query.shape[0]),
                "selected_query_indices": [int(i) for i in selected_query_idx.tolist()],
                "support_class_distribution": {
                    str(int(cls)): int((y_support == cls).sum()) for cls in sorted(set(y_support.tolist()))
                },
                "query_class_distribution_selected": {
                    str(int(cls)): int((y_query_sel == cls).sum()) for cls in sorted(set(y_query_sel.tolist()))
                },
            }
        )

    if signal_length is None:
        raise ValueError(f"Nenhum episódio exportado em {manifest_path}")

    support_all = np.concatenate(support_chunks, axis=0).astype(np.float32)
    support_labels_all = np.concatenate(support_label_chunks, axis=0).astype(np.uint8)
    query_all = np.concatenate(query_chunks, axis=0).astype(np.float32)
    query_labels_all = np.concatenate(query_label_chunks, axis=0).astype(np.uint8)

    export_named_arrays_header(
        arrays=[
            ("ecg_bench_record_ids", np.asarray(record_ids, dtype=np.uint32), "uint32_t"),
            ("ecg_bench_support_offsets", np.asarray(support_offsets, dtype=np.uint32), "uint32_t"),
            ("ecg_bench_support_counts", np.asarray(support_counts, dtype=np.uint32), "uint32_t"),
            ("ecg_bench_query_offsets", np.asarray(query_offsets, dtype=np.uint32), "uint32_t"),
            ("ecg_bench_query_counts", np.asarray(query_counts, dtype=np.uint32), "uint32_t"),
            ("ecg_bench_support", support_all, "float"),
            ("ecg_bench_support_labels", support_labels_all, "uint8_t"),
            ("ecg_bench_query", query_all, "float"),
            ("ecg_bench_query_labels", query_labels_all, "uint8_t"),
        ],
        macros={
            "ECG_BENCH_SIGNAL_LENGTH": signal_length,
            "ECG_BENCH_EPISODES": len(record_ids),
            "ECG_BENCH_TOTAL_SUPPORT_SAMPLES": support_all.shape[0],
            "ECG_BENCH_TOTAL_QUERY_SAMPLES": query_all.shape[0],
            "ECG_BENCH_MAX_SUPPORT_SAMPLES": max_support,
            "ECG_BENCH_MAX_QUERY_SAMPLES": max_query,
            "ECG_BENCH_SHOTS": int(manifest["shots"]),
        },
        out_path=args.out,
    )

    raw_signal_bytes = int((support_all.size + query_all.size) * 4)
    label_bytes = int(support_labels_all.size + query_labels_all.size)
    metadata_bytes = int((len(record_ids) * 5) * 4)
    save_json(
        {
            "manifest": str(manifest_path),
            "header_path": str(Path(args.out)),
            "shots": int(manifest["shots"]),
            "protocol": manifest["protocol"],
            "max_query_samples_per_record": int(args.max_query_samples_per_record),
            "seed": int(args.seed),
            "episodes": episodes,
            "num_episodes": len(record_ids),
            "num_total_support_samples": int(support_all.shape[0]),
            "num_total_query_samples": int(query_all.shape[0]),
            "signal_length": int(signal_length),
            "estimated_flash_bytes": {
                "raw_signals": raw_signal_bytes,
                "labels": label_bytes,
                "metadata": metadata_bytes,
                "total": raw_signal_bytes + label_bytes + metadata_bytes,
            },
        },
        args.summary_out,
    )

    print("Header de benchmark exportado para", args.out)
    print("Resumo salvo em", args.summary_out)


if __name__ == "__main__":
    main()
