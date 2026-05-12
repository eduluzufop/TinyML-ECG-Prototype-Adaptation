#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.export.export_headers import export_named_arrays_header


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", default="firmware/psoc6/generated/demo_replay.h")
    ap.add_argument("--max-query-samples", type=int, default=16)
    args = ap.parse_args()

    x_support, y_support = load_npz(args.support)
    x_query, y_query = load_npz(args.query)
    max_query = min(int(args.max_query_samples), len(y_query))
    x_query = x_query[:max_query]
    y_query = y_query[:max_query]

    export_named_arrays_header(
        arrays=[
            ("ecg_demo_support", x_support[:, 0, :], "float"),
            ("ecg_demo_support_labels", y_support, "uint8_t"),
            ("ecg_demo_query", x_query[:, 0, :], "float"),
            ("ecg_demo_query_labels", y_query, "uint8_t"),
        ],
        macros={
            "ECG_DEMO_SIGNAL_LENGTH": x_support.shape[-1],
            "ECG_DEMO_SUPPORT_SAMPLES": len(y_support),
            "ECG_DEMO_QUERY_SAMPLES": len(y_query),
        },
        out_path=args.out,
    )
    print("Header de replay demo exportado para", args.out)


if __name__ == "__main__":
    main()
