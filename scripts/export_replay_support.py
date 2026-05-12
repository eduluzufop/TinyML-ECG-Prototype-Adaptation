#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.dataset.mitbih import infer_metadata_path, load_dataset_metadata, load_processed_npz
from src.dataset.splits import build_de_chazal_interpatient_split
from src.export.replay_packager import save_replay_pack
from src.personalization.protocols import few_shot_feasible, sample_few_shot
from src.utils.config import load_yaml
from src.utils.io import save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--record", action="append", default=[])
    ap.add_argument("--out_dir", default="firmware/psoc6/generated/replay_support")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data = load_processed_npz(cfg["dataset_path"])
    metadata = load_dataset_metadata(cfg.get("metadata_path", infer_metadata_path(cfg["dataset_path"])))
    x, y = data["x"], data["y"]
    records = data["records"]
    split = build_de_chazal_interpatient_split(records, seed=int(cfg.get("seed", 42)))
    target_records = args.record or list(split.test_records)
    required_classes = sorted(set(y.tolist()))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "shots": int(cfg["shots"]),
        "protocol": metadata.get("split_protocol", "de_chazal_interpatient_ds1_ds2"),
        "exported_records": [],
        "skipped_records": [],
    }

    for record in target_records:
        mask = records == record
        x_target, y_target = x[mask], y[mask]
        ok, reason = few_shot_feasible(
            y_target,
            shots=int(cfg["shots"]),
            required_classes=np.asarray(required_classes),
        )
        if not ok:
            manifest["skipped_records"].append({"record": str(record), "reason": reason})
            continue

        x_sup, y_sup, x_q, y_q = sample_few_shot(
            x_target,
            y_target,
            shots=int(cfg["shots"]),
            seed=int(cfg.get("seed", 42)),
            required_classes=np.asarray(required_classes),
        )
        record_dir = out_dir / str(record)
        save_replay_pack(x_sup, y_sup, str(record_dir / "support.npz"))
        save_replay_pack(x_q, y_q, str(record_dir / "query.npz"))
        manifest["exported_records"].append(
            {
                "record": str(record),
                "support_path": str(record_dir / "support.npz"),
                "query_path": str(record_dir / "query.npz"),
                "num_support_samples": int(len(y_sup)),
                "num_query_samples": int(len(y_q)),
            }
        )

    save_json(manifest, out_dir / "manifest.json")
    print("Support/query exportados para firmware em", out_dir)


if __name__ == "__main__":
    main()
