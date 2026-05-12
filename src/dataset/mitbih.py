from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_processed_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def infer_metadata_path(dataset_path: str | Path) -> Path:
    dataset_path = Path(dataset_path)
    return dataset_path.parent.parent / "metadata" / f"{dataset_path.stem}_metadata.json"


def load_dataset_metadata(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
