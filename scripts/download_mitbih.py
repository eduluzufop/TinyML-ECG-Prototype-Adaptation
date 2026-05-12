#!/usr/bin/env python
from __future__ import annotations

import argparse
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from src.dataset.splits import DE_CHAZAL_ALL_RECORDS

PHYSIONET_MITDB_BASE_URL = "https://physionet.org/files/mitdb/1.0.0"
MITBIH_REQUIRED_EXTENSIONS = ("atr", "dat", "hea")


def download_file(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp_path: Path | None = None
        try:
            with urlopen(url, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    tmp_file.write(response.read())
            tmp_path.replace(destination)
            return
        except (TimeoutError, URLError, OSError) as exc:
            last_error = exc
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                time.sleep(min(5 * attempt, 15))
    assert last_error is not None
    raise last_error


def record_is_complete(out_dir: Path, record: str) -> bool:
    return all((out_dir / f"{record}.{ext}").exists() for ext in MITBIH_REQUIRED_EXTENSIONS)


def download_record(record: str, out_dir: Path) -> str:
    if record_is_complete(out_dir, record):
        return f"[skip] {record}"

    for ext in MITBIH_REQUIRED_EXTENSIONS:
        target = out_dir / f"{record}.{ext}"
        if target.exists():
            continue
        download_file(f"{PHYSIONET_MITDB_BASE_URL}/{record}.{ext}", target)
    return f"[ok] {record}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out) / "mitbih"
    out.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers))
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_record, rec, out): rec for rec in DE_CHAZAL_ALL_RECORDS}
        for future in as_completed(futures):
            record = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append(record)
                print(f"[fail] {record}: {exc}", flush=True)
    if failures:
        raise RuntimeError(f"Falha ao baixar os registros: {', '.join(sorted(failures))}")
    print("Download MIT-BIH concluido em", out)


if __name__ == "__main__":
    main()
