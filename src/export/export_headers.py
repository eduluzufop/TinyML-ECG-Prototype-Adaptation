from __future__ import annotations

from pathlib import Path

import numpy as np


def numpy_to_c_array(arr: np.ndarray, name: str) -> str:
    flat = ", ".join(f"{float(x):.6f}f" for x in arr.flatten())
    return f"static const float {name}[{arr.size}] = {{{flat}}};\n"


def export_header(arr: np.ndarray, name: str, out_path: str) -> None:
    header = "#pragma once\n\n" + numpy_to_c_array(arr, name)
    Path(out_path).write_text(header, encoding="utf-8")


def numpy_to_c_array_typed(arr: np.ndarray, name: str, c_type: str) -> str:
    flat = arr.flatten()
    if c_type == "float":
        values = ", ".join(f"{float(x):.6f}f" for x in flat)
    elif c_type == "uint8_t":
        values = ", ".join(str(int(x)) for x in flat)
    elif c_type == "uint32_t":
        values = ", ".join(f"{int(x)}u" for x in flat)
    else:
        raise ValueError(f"Unsupported c_type: {c_type}")
    return f"static const {c_type} {name}[{arr.size}] = {{{values}}};\n"


def export_named_arrays_header(
    arrays: list[tuple[str, np.ndarray, str]],
    out_path: str | Path,
    macros: dict[str, int] | None = None,
) -> None:
    lines = ["#pragma once", "#include <stdint.h>", ""]
    if macros:
        for key, value in macros.items():
            lines.append(f"#define {key} {int(value)}")
        lines.append("")
    for name, arr, c_type in arrays:
        lines.append(numpy_to_c_array_typed(arr, name, c_type).rstrip())
        lines.append("")
    Path(out_path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
