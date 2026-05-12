# PSoC 6 Firmware

## Requirements

- [ModusToolbox 3.x](https://www.infineon.com/cms/en/design-support/tools/sdk/modustoolbox-software/) (free, Linux / macOS / Windows)
- Board: **CY8CPROTO-063-BLE**
- BSP: `APP_CY8CPROTO-063-BLE` (fetched automatically by `make getlibs`)
- GCC Arm toolchain and OpenOCD (bundled with ModusToolbox)

## Build (CLI)

```bash
cd firmware/psoc6

# First build only: fetch BSP and middleware (~200 MB, one-time)
make getlibs

# Compile Release image
make build CONFIG=Release
```

## Flash

```bash
make program CONFIG=Release
```

## UART output

Connect to the KitProg3 virtual COM port at **115 200 baud**.  
The benchmark firmware prints one line per DS2 episode plus an aggregate summary:

```
BATCH rec=100 proto_f1=0.812 sgd_f1=0.706 pre_f1=0.634
BATCH rec=103 proto_f1=0.779 sgd_f1=0.658 pre_f1=0.611
...
BATCH_AGG proto_f1=0.798 sgd_f1=0.682 pre_f1=0.611 n_records=18
```

## Two firmware modes

| Mode | Source | Description |
|---|---|---|
| **Benchmark** | `app/source/ecg_head_adaptation_app.c` | Replays all embedded DS2 episodes and reports aggregate macro-F1 — used for the paper hardware results |
| **Demo (ecgDEMO)** | `app/source/demo_ecg_app.c` | Streams beat-by-beat JSON over UART for the live UI demo |

## Regenerating embedded headers

The pre-generated headers for the paper's tiny backbone are already in `generated/`.  
To regenerate after retraining, run Steps 3–6 from the [root README](../../README.md):

```bash
# From repository root
export PYTHONPATH=.
python scripts/export_model.py --checkpoint artifacts/checkpoints/baseline.pt
python scripts/export_firmware_benchmark_data.py
cd firmware/psoc6 && make build CONFIG=Release && make program CONFIG=Release
```

## Generated headers

| File | Produced by |
|---|---|
| `generated/model_params.h` | `scripts/export_model.py` |
| `generated/head_weight.h` | `scripts/export_model.py` |
| `generated/head_bias.h` | `scripts/export_model.py` |
| `generated/benchmark_replay.h` | `scripts/export_firmware_benchmark_data.py` |
| `generated/demo_replay.h` | `scripts/export_firmware_demo_data.py` |

## Current status

- On-device head adaptation (prototype + linear SGD) is fully implemented and validated.
- The backbone runs as a hand-optimised native-C function derived from exported float32 weights.
- TF Lite for Microcontrollers (TFLM) integration is prepared in the MTB workspace (`mtb/`) but not yet benchmarked; the validated inference path is the native-C runtime.
- On-device backbone training is not implemented (frozen backbone is the intended deployment mode).
