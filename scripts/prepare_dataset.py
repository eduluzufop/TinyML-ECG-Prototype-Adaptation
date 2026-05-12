#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.dataset.beat_extraction import extract_beat
from src.dataset.label_map import map_label
from src.dataset.splits import DE_CHAZAL_ALL_RECORDS, DE_CHAZAL_DS1_RECORDS, build_de_chazal_interpatient_split
from src.utils.config import load_yaml
from src.utils.io import save_json

try:
    import wfdb
except Exception:
    wfdb = None


def build_synthetic_if_missing(task: str, window_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(-1.0, 1.0, window_size, dtype=np.float32)
    gaussian_peak = np.exp(-20.0 * np.square(time_axis)).astype(np.float32)
    all_records = list(DE_CHAZAL_ALL_RECORDS)
    num_classes = 2 if task == "binary" else 3
    class_symbols = ["N", "A", "V"] if task == "three_class" else ["N", "V"]

    x, y, records, annotation_samples, symbols = [], [], [], [], []
    for record_idx, record in enumerate(all_records):
        domain_shift = 0.15 if record not in DE_CHAZAL_DS1_RECORDS else 0.0
        for beat_idx in range(96):
            label = beat_idx % num_classes
            base = 0.08 * np.sin((3.0 + 0.05 * record_idx) * np.pi * time_axis)
            base += rng.normal(0.0, 0.03 + domain_shift, size=window_size).astype(np.float32)
            if label == 0:
                wave = base + 1.7 * gaussian_peak
            elif label == 1:
                wave = base - 1.0 * gaussian_peak + 0.12 * np.sign(time_axis)
            else:
                wave = base + 0.7 * np.sin(6.0 * np.pi * time_axis) + 0.5 * gaussian_peak
            x.append(wave.astype(np.float32))
            y.append(label)
            records.append(record)
            annotation_samples.append(beat_idx)
            symbols.append(class_symbols[label])

    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(records),
        np.asarray(annotation_samples, dtype=np.int64),
        np.asarray(symbols),
        "synthetic_fallback",
    )


def _record_is_available(raw_dir: Path, record: str) -> bool:
    return all((raw_dir / f"{record}.{ext}").exists() for ext in ("dat", "hea", "atr"))


def _select_signal_channel(record: "wfdb.Record", preferred_leads: list[str]) -> np.ndarray:
    for lead in preferred_leads:
        if lead in record.sig_name:
            return record.p_signal[:, record.sig_name.index(lead)].astype(np.float32)
    return record.p_signal[:, 0].astype(np.float32)


def load_beats_from_raw_dataset(
    raw_dir: Path,
    *,
    task: str,
    pre_samples: int,
    post_samples: int,
    preferred_leads: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if wfdb is None:
        raise RuntimeError("wfdb nao esta disponivel, mas os arquivos MIT-BIH brutos existem.")

    x, y, records, annotation_samples, symbols = [], [], [], [], []
    for record_name in DE_CHAZAL_ALL_RECORDS:
        base_path = raw_dir / record_name
        record = wfdb.rdrecord(str(base_path))
        ann = wfdb.rdann(str(base_path), "atr")
        signal = _select_signal_channel(record, preferred_leads)
        for sample, symbol in zip(ann.sample, ann.symbol):
            label = map_label(symbol, task=task)
            if label is None:
                continue
            beat = extract_beat(signal, r_index=int(sample), pre=pre_samples, post=post_samples)
            if beat is None:
                continue
            x.append(beat)
            y.append(label)
            records.append(record_name)
            annotation_samples.append(int(sample))
            symbols.append(symbol)

    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(records),
        np.asarray(annotation_samples, dtype=np.int64),
        np.asarray(symbols),
        "mitbih_wfdb",
    )


def build_class_distribution(y: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def build_record_distribution(records: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(records.astype(str), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    raw_dir = Path(cfg["raw_dir"])
    out = Path(cfg["processed_path"])
    metadata_path = Path(cfg.get("metadata_path", out.parent.parent / "metadata" / f"{out.stem}_metadata.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    task = cfg.get("task", "binary")
    pre_samples = int(cfg["window"]["pre_samples"])
    post_samples = int(cfg["window"]["post_samples"])
    window_size = pre_samples + post_samples
    seed = int(cfg.get("seed", 42))
    preferred_leads = list(cfg.get("preferred_leads", ["MLII", "ML2", "V5", "II"]))

    available_records = [record for record in DE_CHAZAL_ALL_RECORDS if _record_is_available(raw_dir, record)]
    if available_records:
        missing_records = sorted(set(DE_CHAZAL_ALL_RECORDS) - set(available_records))
        if missing_records:
            raise RuntimeError(
                "Dataset MIT-BIH incompleto para o protocolo DS1/DS2. Registros ausentes: "
                + ", ".join(missing_records)
            )
        x, y, records, annotation_samples, symbols, source = load_beats_from_raw_dataset(
            raw_dir,
            task=task,
            pre_samples=pre_samples,
            post_samples=post_samples,
            preferred_leads=preferred_leads,
        )
    else:
        x, y, records, annotation_samples, symbols, source = build_synthetic_if_missing(
            task=task,
            window_size=window_size,
            seed=seed,
        )

    # valida extração/normalizacao no formato final
    beat = extract_beat(x[0], r_index=pre_samples, pre=pre_samples, post=post_samples)
    assert beat is not None and len(beat) == window_size
    _ = map_label("N", task=task)

    x = x[:, None, :]  # [N,1,T]
    split = build_de_chazal_interpatient_split(
        records,
        seed=seed,
        val_fraction=float(cfg.get("val_fraction", 0.1)),
    )
    split_group = np.where(np.isin(records.astype(str), DE_CHAZAL_DS1_RECORDS), "DS1", "DS2")

    np.savez(
        out,
        x=x,
        y=y,
        records=records,
        annotation_samples=annotation_samples,
        symbols=symbols,
        split_group=split_group,
    )

    metadata = {
        "dataset_name": cfg.get("dataset_name", "mitbih"),
        "source": source,
        "processed_path": str(out),
        "task": task,
        "normalization": cfg.get("normalization", "zscore_per_beat"),
        "window": {"pre_samples": pre_samples, "post_samples": post_samples, "window_size": window_size},
        "split_protocol": "de_chazal_interpatient_ds1_ds2",
        "records": list(DE_CHAZAL_ALL_RECORDS),
        "class_distribution": build_class_distribution(y),
        "record_distribution": build_record_distribution(records),
        "split": split.to_metadata(),
    }
    save_json(metadata, metadata_path)
    print(f"Dataset preparado: {out} | shape={x.shape} | source={source}")
    print(f"Metadata salva em: {metadata_path}")


if __name__ == "__main__":
    main()
