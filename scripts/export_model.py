#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.export.export_headers import export_header, export_named_arrays_header
from src.models.cnn1d_backbone import build_model_from_kwargs, normalize_model_kwargs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="firmware/psoc6/generated")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_kwargs = normalize_model_kwargs(ckpt.get("model_kwargs", {"embedding_dim": int(ckpt.get("embedding_dim", 32))}))
    model = build_model_from_kwargs(num_classes=int(ckpt["num_classes"]), model_kwargs=model_kwargs)
    model.load_state_dict(ckpt["state_dict"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w = model.head.weight.detach().cpu().numpy().astype(np.float32)
    b = model.head.bias.detach().cpu().numpy().astype(np.float32)
    conv1_w = model.backbone.features[0].weight.detach().cpu().numpy().astype(np.float32)
    conv1_b = model.backbone.features[0].bias.detach().cpu().numpy().astype(np.float32)
    conv2_w = model.backbone.features[3].weight.detach().cpu().numpy().astype(np.float32)
    conv2_b = model.backbone.features[3].bias.detach().cpu().numpy().astype(np.float32)
    proj_w = model.backbone.proj.weight.detach().cpu().numpy().astype(np.float32)
    proj_b = model.backbone.proj.bias.detach().cpu().numpy().astype(np.float32)

    export_header(w, "head_weight", str(out_dir / "head_weight.h"))
    export_header(b, "head_bias", str(out_dir / "head_bias.h"))
    export_named_arrays_header(
        arrays=[
            ("ecg_conv1_weight", conv1_w, "float"),
            ("ecg_conv1_bias", conv1_b, "float"),
            ("ecg_conv2_weight", conv2_w, "float"),
            ("ecg_conv2_bias", conv2_b, "float"),
            ("ecg_proj_weight", proj_w, "float"),
            ("ecg_proj_bias", proj_b, "float"),
            ("ecg_head_weight", w, "float"),
            ("ecg_head_bias", b, "float"),
        ],
        macros={
            "ECG_INPUT_LENGTH": 200,
            "ECG_CONV1_OUT_CHANNELS": conv1_w.shape[0],
            "ECG_CONV1_IN_CHANNELS": conv1_w.shape[1],
            "ECG_CONV1_KERNEL_SIZE": conv1_w.shape[2],
            "ECG_CONV2_OUT_CHANNELS": conv2_w.shape[0],
            "ECG_CONV2_IN_CHANNELS": conv2_w.shape[1],
            "ECG_CONV2_KERNEL_SIZE": conv2_w.shape[2],
            "ECG_PROJ_IN_DIM": proj_w.shape[1],
            "ECG_PROJ_OUT_DIM": proj_w.shape[0],
            "ECG_NUM_CLASSES": w.shape[0],
        },
        out_path=out_dir / "model_params.h",
    )
    np.savez(
        out_dir / "head_params.npz",
        weight=w,
        bias=b,
        conv1_weight=conv1_w,
        conv1_bias=conv1_b,
        conv2_weight=conv2_w,
        conv2_bias=conv2_b,
        proj_weight=proj_w,
        proj_bias=proj_b,
    )
    print("Artefatos do backbone e da cabeça exportados em", out_dir)


if __name__ == "__main__":
    main()
