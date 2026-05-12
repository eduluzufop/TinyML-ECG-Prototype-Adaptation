from __future__ import annotations

from src.personalization.adaptation_metrics import gain


def build_gain_report(pre: dict[str, float], post: dict[str, float]) -> dict[str, float]:
    acc = gain(pre["accuracy"], post["accuracy"])
    f1 = gain(pre["f1_macro"], post["f1_macro"])
    return {
        "acc_abs_gain": acc["absolute_gain"],
        "acc_rel_gain": acc["relative_gain"],
        "f1_abs_gain": f1["absolute_gain"],
        "f1_rel_gain": f1["relative_gain"],
    }
