from __future__ import annotations

from pathlib import Path


def export_placeholder_tflite(output_path: str) -> None:
    Path(output_path).write_bytes(b"TFLITE_PLACEHOLDER")
