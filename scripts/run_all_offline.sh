#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

python3 scripts/download_mitbih.py --out data/raw
python3 scripts/prepare_dataset.py --config configs/dataset/mitbih_binary.yaml
python3 scripts/train_backbone.py --config configs/training/baseline.yaml
python3 scripts/evaluate_pre_adaptation.py --config configs/personalization/protocol_1shot.yaml
python3 scripts/evaluate_pre_adaptation.py --config configs/personalization/protocol_5shot.yaml
python3 scripts/evaluate_pre_adaptation.py --config configs/personalization/protocol_10shot.yaml
python3 scripts/export_embedding_dataset.py --checkpoint artifacts/checkpoints/baseline.pt --dataset data/processed/mitbih_binary.npz --out artifacts/embeddings/embeddings.npz
python3 scripts/export_model.py --checkpoint artifacts/checkpoints/baseline.pt
python3 scripts/export_replay_support.py --config configs/personalization/protocol_1shot.yaml --out_dir firmware/psoc6/generated/replay_support_1shot
python3 scripts/export_replay_support.py --config configs/personalization/protocol_5shot.yaml --out_dir firmware/psoc6/generated/replay_support
python3 scripts/export_replay_support.py --config configs/personalization/protocol_10shot.yaml --out_dir firmware/psoc6/generated/replay_support_10shot
python3 scripts/export_firmware_demo_data.py --support firmware/psoc6/generated/replay_support/100/support.npz --query firmware/psoc6/generated/replay_support/100/query.npz --out firmware/psoc6/generated/demo_replay.h --max-query-samples 16
python3 scripts/summarize_results.py
