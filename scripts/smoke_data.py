#!/usr/bin/env python3
"""Decode ARCTIC/HOT3D manifest rows and audit the HandPrism sample contract."""

from __future__ import annotations

import argparse
from dataclasses import fields
import gc
import json
from pathlib import Path
import time

import torch

from dreamhand.data import DreamHandWindowDataset
from dreamhand.data.contract import DreamHandSample, validate_sample


def audit_sample(sample: DreamHandSample, elapsed: float) -> dict[str, object]:
    validate_sample(sample)
    nonfinite: dict[str, int] = {}
    for field in fields(sample):
        value = getattr(sample, field.name)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            count = int((~torch.isfinite(value)).sum())
            if count:
                nonfinite[field.name] = count
    if nonfinite:
        raise RuntimeError(f"non-finite sample tensors: {nonfinite}")
    if float(sample.video.min()) < -1.001 or float(sample.video.max()) > 1.001:
        raise RuntimeError("video normalization is outside [-1,1]")
    return {
        "dataset": sample.dataset,
        "recording_id": sample.recording_id,
        "frame_start": int(sample.frame_indices[0]),
        "frame_stop": int(sample.frame_indices[-1]) + 1,
        "video_shape": list(sample.video.shape),
        "image_size": sample.image_size.tolist(),
        "intrinsics": sample.intrinsics.tolist(),
        "camera_model": sample.camera_model,
        "gt_ray_shape": list(sample.gt_ray_field.shape)
        if sample.gt_ray_field is not None
        else None,
        "valid_hand_frames": int(sample.valid_hand.sum()),
        "valid_mano_frames": int(sample.valid_mano.sum()),
        "valid_3d_joints": int(sample.valid_joints_3d.sum()),
        "valid_2d_joints": int(sample.valid_joints_2d.sum()),
        "valid_ray": bool(sample.valid_ray),
        "video_min": float(sample.video.min()),
        "video_max": float(sample.video.max()),
        "decode_seconds": elapsed,
        "contract": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/manifests/two_dataset_v2_clean")
    )
    parser.add_argument("--mano-model", type=Path, default=Path("assets/body_models/mano"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--datasets", nargs="+", choices=("arctic", "hot3d"),
        default=("arctic", "hot3d"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    mano_model = args.mano_model if args.mano_model.is_absolute() else root / args.mano_model
    torch.manual_seed(260820308)
    for name in args.datasets:
        dataset = DreamHandWindowDataset(
            manifest_root / f"{name}_{args.split}.jsonl",
            mano_model_path=mano_model,
            training=args.split == "train",
        )
        if not 0 <= args.index < len(dataset):
            raise IndexError(f"{name} index {args.index} is outside [0,{len(dataset)})")
        started = time.perf_counter()
        sample = dataset[args.index]
        print(json.dumps(audit_sample(sample, time.perf_counter() - started), sort_keys=True))
        del sample, dataset
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
