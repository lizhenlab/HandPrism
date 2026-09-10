#!/usr/bin/env python3
"""Real train-sample geometry and tiny-decoder overfit; never loads Wan/GPU.

This validates engineering paths, NOT full-model convergence or test accuracy.
Only one training recording from each allowed source is decoded; test RGB is
not loaded. Results and small diagnostic tensors are written to a new output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

from dreamhand.camera import project_camera
from dreamhand.config import DecoderConfig
from dreamhand.data import DreamHandWindowDataset, collate_samples
from dreamhand.data.policy import SUPPORTED_DATASETS
from dreamhand.decoder import DreamHandDecoder
from dreamhand.ray import (
    kfree_bearings,
    bearings_from_intrinsics,
    mixed_pnp,
    project_kfree,
    sample_ray_bearings,
)
from dreamhand.training import target_from_batch
from dreamhand.completion import require_finite_json
from scripts.train import load_config, manifest_report
from scripts.check_readiness import audit_manifests, audit_config


def features_from_frame(video: torch.Tensor) -> torch.Tensor:
    """RGB/coordinate/random fixed features for a tiny decoder diagnostic only."""
    rgb = F.interpolate(
        video[:, :, video.shape[2] // 2], (12, 16), mode="bilinear", align_corners=False
    )
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 12), torch.linspace(-1, 1, 16), indexing="ij")
    positional = torch.stack([xx, yy, xx * yy, xx.square(), yy.square()], -1)
    raw = torch.cat([rgb.permute(0, 2, 3, 1), positional[None]], -1)
    return (raw @ torch.randn(8, 32))[..., :][:, None].detach()


def render_overfit(path: Path, frame, truth, before, after, valid):
    pixels = frame.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    original = Image.fromarray(pixels)
    width, height = original.size
    canvas = Image.new("RGB", (width * 3, height + 32))
    for index, (label, points) in enumerate(
        (
            ("Ground truth", truth),
            ("Tiny decoder: before", before),
            ("Tiny decoder: 200 steps", after),
        )
    ):
        panel = original.copy()
        draw = ImageDraw.Draw(panel)
        for side, color in enumerate(((0, 255, 160), (255, 100, 210))):
            for joint in range(21):
                if not valid[side, joint]:
                    continue
                x, y = (points[side, joint] * points.new_tensor([width, height])).tolist()
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        canvas.paste(panel, (index * width, 32))
        ImageDraw.Draw(canvas).text((index * width + 8, 8), label, fill="white")
    canvas.save(path)


def run(root: Path, output: Path, iterations: int, *, architecture: str) -> dict:
    from dreamhand.architectures import require_config_architecture
    from scripts.train import solver_config_from_json

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit must not initialize CUDA")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(43)
    stem = architecture.replace("-", "_")
    config = load_config(root / f"configs/{stem}_standard.json", architecture=architecture)
    require_config_architecture(config, architecture)
    solver_config = solver_config_from_json(config)
    manifest_root = root / config["manifests"]
    checks = {}
    audit_manifests(manifest_root, SUPPORTED_DATASETS, checks, True)
    for name in (f"{stem}_standard.json", f"{stem}_kfree.json"):
        audit_config(root, root / "configs" / name, manifest_root, SUPPORTED_DATASETS, checks)
    if not all(check["pass"] for check in checks.values()):
        raise RuntimeError(f"readiness failed: {checks}")
    report = {
        "architecture": architecture,
        "scope": "CPU real train-sample geometry and tiny-decoder diagnostics; not Wan training or test accuracy",
        "config_and_manifest_checks": checks,
        "split": manifest_report(manifest_root, SUPPORTED_DATASETS),
        "datasets": {},
        "iterations": iterations,
    }
    for name in SUPPORTED_DATASETS:
        started = time.perf_counter()
        dataset = DreamHandWindowDataset(
            manifest_root / f"{name}_train.jsonl",
            mano_model_path=root / config["mano_model"],
            training=True,
        )
        sample = dataset[(0, 43)]
        batch = collate_samples([sample])
        targets = target_from_batch(batch, 32, 32)
        camera_kwargs = {
            key: batch[key] for key in ("camera_model", "camera_parameters", "source_image_size")
        }

        def calibrated_projector(points):
            uv = project_camera(
                points,
                batch["intrinsics"],
                batch["image_size"],
                batch["distortion"],
                **camera_kwargs,
            )
            return uv, torch.isfinite(uv).all(-1) & (points[..., 2] > 0)

        anchors = batch["joints_2d"]
        depth = batch["translation"][..., 2:].clamp_min(0.05).log()
        bearings = (
            bearings_from_intrinsics(
                anchors, batch["intrinsics"], batch["image_size"], batch["distortion"]
            )
            if batch["gt_ray_field"] is None
            else sample_ray_bearings(batch["gt_ray_field"], anchors)
        )
        standard = mixed_pnp(
            batch["joints_root"],
            anchors,
            depth,
            bearings,
            batch["image_size"],
            solver_config,
            projector=calibrated_projector,
        )
        k_bearings, fit = kfree_bearings(targets.ray_field, anchors, solver_config)
        kfree = mixed_pnp(
            batch["joints_root"],
            anchors,
            depth,
            k_bearings,
            batch["image_size"],
            solver_config,
            projector=lambda p: project_kfree(p, fit, targets.ray_field, anchors),
        )
        eligible = batch["valid_hand"] & (standard.vote_count >= 6)
        geometry = {}
        for solver, result in (("standard", standard), ("kfree_gt_ray_oracle", kfree)):
            valid = eligible & result.solved
            geometry[solver] = {
                "eligible_hand_frames": int(eligible.sum()),
                "solved_eligible_hand_frames": int(valid.sum()),
                "pnp_residual_kind": result.residual_kind,
                "mean_pnp_residual_solved": float(result.rms_pixels[valid].mean())
                if valid.any()
                else None,
                "translation_error_mm_solved": float(
                    (result.translation - batch["translation"])[valid].norm(dim=-1).mean() * 1000
                )
                if valid.any()
                else None,
            }
        assert geometry["standard"]["eligible_hand_frames"] > 0
        assert geometry["standard"]["solved_eligible_hand_frames"] > 0
        geometry["oracle_pinhole_fit_accepted"] = bool(fit.valid[0])
        geometry["oracle_fit_rms_normalized"] = float(fit.rms_normalized[0])
        model = DreamHandDecoder(
            DecoderConfig(feature_dim=32, hidden_dim=64, layers=2, heads=4, ffn_dim=128),
            architecture=architecture,
        )
        features = features_from_frame(batch["video"])
        ray = target_from_batch(batch, 12, 16).ray_field
        middle = sample.video.shape[1] // 2
        truth_2d = batch["joints_2d"][:, middle : middle + 1]
        truth_3d = batch["joints_root"][:, middle : middle + 1]
        mask_2d = batch["valid_joints_2d"][:, middle : middle + 1]
        mask_3d = batch["valid_joints_3d"][:, middle : middle + 1]
        assert mask_2d.any() and mask_3d.any()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0)
        measurements = []
        with (output / f"overfit_{name}.jsonl").open("w") as log:
            for step in range(iterations + 1):
                decoded = model(features, ray, 1)
                if step == 0:
                    initial_anchors = decoded.anchors_2d.detach().clone()
                delta_2d = decoded.anchors_2d - truth_2d
                delta_3d = decoded.joints_root_direct - truth_3d
                loss_2d = delta_2d.square().sum(-1)[mask_2d].mean()
                loss_3d = delta_3d.square().sum(-1)[mask_3d].mean()
                loss = loss_2d + 10 * loss_3d
                record = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "anchors_epe_px": float(
                        (delta_2d * batch["image_size"][:, None, None, None, [1, 0]])
                        .norm(dim=-1)[mask_2d]
                        .detach()
                        .mean()
                    ),
                    "direct_root_mpjpe_mm": float(
                        delta_3d.norm(dim=-1)[mask_3d].detach().mean() * 1000
                    ),
                }
                require_finite_json(record)
                log.write(json.dumps(record) + "\n")
                if step in (0, iterations):
                    measurements.append(record)
                if step == iterations:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert all(
                    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
                )
                optimizer.step()
        assert measurements[-1]["loss"] < measurements[0]["loss"] * 0.5
        assert measurements[-1]["anchors_epe_px"] < measurements[0]["anchors_epe_px"]
        render_overfit(
            output / f"overfit_{name}.png",
            batch["video"][0, :, middle],
            truth_2d[0, 0],
            initial_anchors[0, 0],
            decoded.anchors_2d.detach()[0, 0],
            mask_2d[0, 0],
        )
        torch.save(
            {
                "video": batch["video"][:, :, middle],
                "anchors_initial": initial_anchors,
                "anchors_final": decoded.anchors_2d.detach(),
                "gt_2d": truth_2d,
                "valid_2d": mask_2d,
            },
            output / f"overlay_{name}.pt",
        )
        report["datasets"][name] = {
            "recording_id": sample.recording_id,
            "camera_model": sample.camera_model,
            "geometry": geometry,
            "tiny_overfit": measurements,
            "seconds": time.perf_counter() - started,
        }
        print(json.dumps({"dataset": name, **report["datasets"][name]}), flush=True)
    assert not torch.cuda.is_initialized()
    require_finite_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    from dreamhand.architectures import add_architecture_argument

    parser = argparse.ArgumentParser(description=__doc__)
    add_architecture_argument(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.output, args.iterations, architecture=args.architecture)
