from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DE_CHAZAL_DS1_RECORDS: tuple[str, ...] = (
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
)

DE_CHAZAL_DS2_RECORDS: tuple[str, ...] = (
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
)

DE_CHAZAL_ALL_RECORDS: tuple[str, ...] = DE_CHAZAL_DS1_RECORDS + DE_CHAZAL_DS2_RECORDS


@dataclass
class DatasetSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_records: tuple[str, ...]
    val_records: tuple[str, ...]
    test_records: tuple[str, ...]
    split_name: str = "de_chazal_interpatient"

    def to_metadata(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "train_records": list(self.train_records),
            "val_records": list(self.val_records),
            "test_records": list(self.test_records),
            "train_size": int(len(self.train_idx)),
            "val_size": int(len(self.val_idx)),
            "test_size": int(len(self.test_idx)),
        }


def _unique_ordered(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(v) for v in values))


def get_de_chazal_record_groups() -> dict[str, tuple[str, ...]]:
    return {
        "DS1": DE_CHAZAL_DS1_RECORDS,
        "DS2": DE_CHAZAL_DS2_RECORDS,
        "all": DE_CHAZAL_ALL_RECORDS,
    }


def build_de_chazal_interpatient_split(
    records: np.ndarray,
    seed: int = 42,
    val_fraction: float = 0.1,
) -> DatasetSplit:
    record_array = np.asarray(records).astype(str)
    present_records = set(record_array.tolist())

    ds1_present = [record for record in DE_CHAZAL_DS1_RECORDS if record in present_records]
    ds2_present = [record for record in DE_CHAZAL_DS2_RECORDS if record in present_records]
    if not ds1_present:
        raise ValueError("Nenhum registro DS1 encontrado no dataset processado.")
    if not ds2_present:
        raise ValueError("Nenhum registro DS2 encontrado no dataset processado.")

    rng = np.random.default_rng(seed)
    shuffled_ds1 = ds1_present.copy()
    rng.shuffle(shuffled_ds1)

    if len(shuffled_ds1) == 1 or val_fraction <= 0:
        val_records: list[str] = []
    else:
        n_val_records = max(1, int(round(len(shuffled_ds1) * val_fraction)))
        n_val_records = min(n_val_records, len(shuffled_ds1) - 1)
        val_records = sorted(shuffled_ds1[:n_val_records])

    train_records = sorted(record for record in shuffled_ds1 if record not in set(val_records))
    test_records = sorted(ds2_present)

    train_idx = np.where(np.isin(record_array, train_records))[0]
    val_idx = np.where(np.isin(record_array, val_records))[0]
    test_idx = np.where(np.isin(record_array, test_records))[0]

    return DatasetSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_records=_unique_ordered(train_records),
        val_records=_unique_ordered(val_records),
        test_records=_unique_ordered(test_records),
    )
