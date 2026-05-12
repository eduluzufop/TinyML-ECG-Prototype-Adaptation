from __future__ import annotations

from pathlib import Path

import numpy as np


def save_replay_pack(x: np.ndarray, y: np.ndarray, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, x=x, y=y)
