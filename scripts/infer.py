#!/usr/bin/env python3
"""Run an explicitly selected HandPrism architecture on a prepared RGB clip.

Input NPZ: uint8 RGB video [T,H,W,3], T in {1,5,...,81}, H/W multiples
of 32. Standard additionally needs calibrated intrinsics and camera fields.
No resizing, stereo splitting, camera guessing or architecture inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamhand.architectures import (
    add_architecture_argument, validate_checkpoint_identity, require_legacy_digest,
)
from dreamhand.camera import PINHOLE, FISHEYE624_UPRIGHT
from dreamhand.data.policy import allowed_path
from dreamhand.paths import v3_run_path
from dreamhand.precision import _tensors
from dreamhand.release import load_released_weights
from scripts.evaluate import build_system, save_prediction
from scripts.train import load_config, load_trainable_state, resolve, sha256


def load_clip(path: Path, solver: str) -> dict[str, Any]:
    """Validate a prepared clip on CPU, before loading the backbone or GPU."""
    if solver not in {"standard", "kfree"}:
        raise ValueError("solver must be standard or kfree")
    with np.load(allowed_path(path), allow_pickle=False) as data:
        video = data["video"]
        if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError("video must be uint8 RGB [T,H,W,3]")
        frames, height, width, _ = video.shape
        if not (1 <= frames <= 81 and frames % 4 == 1 and height > 0 and width > 0
                and height % 32 == 0 and width % 32 == 0):
            raise ValueError("require T=1 mod 4 (at most 81), positive H/W multiples of 32")
        batch: dict[str, Any] = {
            "video": torch.from_numpy(video.copy()).permute(3, 0, 1, 2)[None].float() / 127.5 - 1,
            "image_size": torch.tensor([[height, width]], dtype=torch.float32),
            "frame_indices": torch.arange(frames)[None],
            "dataset": "user_rgb_clip",
            "recording_id": [path.stem],
            "intrinsics": None,
            "distortion": None,
            "camera_model": PINHOLE,
            "camera_parameters": None,
            "source_image_size": None,
            "gt_ray_field": None,
        }
        # K-free never consumes supplied GT camera fields, even if present.
        if solver == "kfree":
            return batch

        def numeric(name: str, shape: tuple[int, ...]) -> torch.Tensor:
            if name not in data:
                raise ValueError(f"Standard requires {name}")
            value = np.asarray(data[name])
            if value.shape != shape or not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"{name} must have shape {shape} and numeric dtype")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains nonfinite values")
            return torch.from_numpy(value.astype(np.float32, copy=True))[None]

        batch["intrinsics"] = numeric("intrinsics", (3, 3))
        k = batch["intrinsics"][0]
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not torch.allclose(k[2], k.new_tensor([0, 0, 1])):
            raise ValueError("invalid camera intrinsics")
        camera_model = str(data["camera_model"].item()) if "camera_model" in data else PINHOLE
        if camera_model not in {PINHOLE, FISHEYE624_UPRIGHT}:
            raise ValueError("unknown camera model")
        batch["camera_model"] = camera_model
        if "distortion" in data:
            batch["distortion"] = numeric("distortion", (8,))
        if camera_model == FISHEYE624_UPRIGHT:
            batch["camera_parameters"] = numeric("camera_parameters", (15,))
            batch["source_image_size"] = numeric("source_image_size", (2,))
            batch["gt_ray_field"] = numeric("calibration_ray_field", (height // 32, width // 32, 3))
        elif "calibration_ray_field" in data:
            batch["gt_ray_field"] = numeric("calibration_ray_field", (height // 32, width // 32, 3))
        return batch


@torch.inference_mode()
def predict_clip(system, vae_encoder, batch: dict[str, Any], config: dict[str, Any], device):
    if system.architecture != config["architecture"]:
        raise ValueError("inference model/config architecture mismatch")
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    dtype = torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float16
    video = batch["video"].to(dtype if device.type == "cuda" else torch.float32)
    latent = vae_encoder(video)
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
        output = system(
            latent, target_frames=video.shape[2], solver=config["solver"],
            intrinsics=batch["intrinsics"], image_size=batch["image_size"],
            distortion=batch["distortion"], camera_model=batch["camera_model"],
            calibration_ray_field=batch["gt_ray_field"] if config["solver"] == "standard" else None,
            camera_parameters=batch["camera_parameters"], source_image_size=batch["source_image_size"],
        )
    return output, batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_architecture_argument(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory inside runs/")
    parser.add_argument("--device", default="cuda:0")
    weights_mode = parser.add_mutually_exclusive_group()
    weights_mode.add_argument("--legacy-weights", action="store_true")
    weights_mode.add_argument("--released-weights", action="store_true",
                             help="load SHA-256-pinned Core safetensors from the GitHub release")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(allowed_path(resolve(root, str(args.config))), architecture=args.architecture)
    checkpoint_path = allowed_path(resolve(root, str(args.checkpoint)))
    output_dir = v3_run_path(root, args.output)
    if output_dir.exists():
        raise FileExistsError(f"inference output already exists: {output_dir}")
    digest = sha256(checkpoint_path)
    if args.released_weights:
        checkpoint = load_released_weights(checkpoint_path, config, architecture=args.architecture)
    else:
        if args.legacy_weights:
            require_legacy_digest(args.architecture, config["solver"], digest)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        validate_checkpoint_identity(
            checkpoint, config, architecture=args.architecture,
            legacy_sha256=digest if args.legacy_weights else None,
        )
    batch = load_clip(allowed_path(resolve(root, str(args.input))), config["solver"])
    device = torch.device(args.device)
    dtype = torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float16
    system, vae_encoder = build_system(root, config, device, dtype if device.type == "cuda" else torch.float32)
    load_trainable_state(system, checkpoint["trainable"])
    output, batch = predict_clip(system, vae_encoder, batch, config, device)
    if any(not torch.isfinite(value).all() for value in _tensors(output) if value.is_floating_point()):
        raise ValueError("nonfinite inference output; no prediction was exported")
    output_dir.mkdir(parents=True, exist_ok=False)
    save_prediction(
        output_dir / "prediction.npz", output, batch, int(checkpoint["step"]),
        config["solver"], args.architecture,
    )
    import json
    (output_dir / "provenance.json").write_text(json.dumps({
        "architecture": args.architecture, "implementation_id": config["implementation_id"],
        "solver": config["solver"], "checkpoint_sha256": digest,
        "checkpoint_step": int(checkpoint["step"]), "legacy_weights": args.legacy_weights,
        "released_weights": args.released_weights,
        "input_sha256": sha256(allowed_path(resolve(root, str(args.input)))),
        "scope": "inference only; not training, validation or test metrics",
    }, indent=2) + "\n")
    print(output_dir / "prediction.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
