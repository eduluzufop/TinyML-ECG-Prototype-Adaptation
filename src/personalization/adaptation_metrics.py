from __future__ import annotations


def gain(pre: float, post: float) -> dict[str, float]:
    abs_gain = post - pre
    rel_gain = abs_gain / pre if pre != 0 else 0.0
    return {"absolute_gain": abs_gain, "relative_gain": rel_gain}
