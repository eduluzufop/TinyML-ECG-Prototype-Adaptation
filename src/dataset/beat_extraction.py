from __future__ import annotations

import numpy as np


def extract_beat(signal: np.ndarray, r_index: int, pre: int, post: int) -> np.ndarray | None:
    start = r_index - pre
    end = r_index + post
    if start < 0 or end > len(signal):
        return None
    beat = signal[start:end].astype(np.float32)
    if beat.std() > 1e-8:
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)
    return beat
