from __future__ import annotations

MITBIH_NORMAL = {"N"}
MITBIH_NON_NORMAL = {"L", "R", "A", "V", "/", "f", "F", "j", "a", "E", "J", "S"}


def map_label(symbol: str, task: str = "binary") -> int | None:
    if task == "binary":
        if symbol in MITBIH_NORMAL:
            return 0
        if symbol in MITBIH_NON_NORMAL:
            return 1
        return None
    if task == "three_class":
        if symbol in {"N", "L", "R"}:
            return 0
        if symbol in {"A", "a", "J", "S"}:
            return 1
        if symbol in {"V", "E", "F"}:
            return 2
        return None
    raise ValueError(f"Unsupported task: {task}")
