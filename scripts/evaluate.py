#!/usr/bin/env python3
"""Distributed HandPrism evaluation on the configured ARCTIC/HOT3D split."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from handprism.backbone import (
    WanCleanLatentEncoder,
    WanFrozenVAEEncoder,
    load_official_vae,
    load_official_wan,
)
from handprism.data import HandPrismWindowDataset, collate_samples
from handprism.evaluator import (
    EvaluationAccumulator,
    score_batch,
    validate_metric_result,
)
from handprism.lora import configure_trainable_backbone, inject_wan_lora
from handprism.mano import SmplxMano
from handprism.paths import v3_run_path
from handprism.system import HandPrismSystem
from handprism.fusion_runtime import fusion_config_from_json, dataset_options, fusion_forward_options
from handprism.architectures import (
    add_architecture_argument, architecture_spec, validate_checkpoint_identity, require_legacy_digest,
)
from handprism.data.policy import allowed_path
from handprism.completion import require_finite_json
from handprism.release import load_released_weights
from scripts.train import (
    DistributedEvalSampler,
    load_config,
    load_trainable_state,
    loader_options,
    move_batch,
    resolve,
    sha256,
    solver_config_from_json,
    decoder_config_from_json,
)


def distributed_context() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return rank, world, device


def prediction_name(dataset: str, recording_id: str, frame_start: int) -> str:
    digest = hashlib.sha256(f"{dataset}:{recording_id}:{frame_start}".encode()).hexdigest()[:20]
    return f"{digest}.npz"


def save_prediction(
    path: Path,
    output: Any,
    batch: dict[str, Any],
    checkpoint_step: int,
    solver: str,
    architecture: str,
) -> None:
    metadata = {
        "format": "handprism-segment-prediction",
        "metric_protocol_version": 2,
        "timestamp_source": batch.get("timestamp_source"),
        "dataset": batch["dataset"],
        "recording_id": batch["recording_id"][0],
        "frame_start": int(batch["frame_indices"][0, 0]),
        "frame_stop": int(batch["frame_indices"][0, -1]) + 1,
        "checkpoint_step": checkpoint_step,
        "solver": solver,
        "architecture": architecture,
        "implementation_id": architecture_spec(architecture).implementation_id,
        "pnp_residual_kind": output.pnp.residual_kind,
        "slot_order": ["left", "right"],
        "translation_convention": "translation is wrist_camera; mano_translation is native MANO tau",
    }

    def array(value: torch.Tensor, *, half: bool = False) -> np.ndarray:
        value = value[0].detach().cpu()
        if value.is_floating_point():
            value = value.to(torch.float16 if half else torch.float32)
        return value.numpy()

    payload = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "frame_indices": array(batch["frame_indices"]),
        "global_rotation": array(output.decoder.global_rotation),
        "articulation": array(output.decoder.articulation),
        "betas": array(output.decoder.betas),
        "translation": array(output.pnp.translation),
        "wrist_camera": array(output.pnp.translation),
        "mano_translation": array(output.mano_translation),
        "joints_root_mano": array(output.joints_root_mano),
        "joints_camera": array(output.joints_camera),
        "anchors_2d": array(output.decoder.anchors_2d),
        "existence_score": array(output.decoder.existence_logits.sigmoid()),
        "visibility_score": array(output.decoder.visibility_logits.sigmoid()),
        "pnp_solved": array(output.pnp.solved),
        "pnp_votes": array(output.pnp.vote_count),
        "pnp_rms_pixels": array(output.pnp.rms_pixels),
        "pnp_projection_valid": array(output.pnp.projection_valid),
        "pnp_failure_code": array(output.pnp.failure_code),
        "joints_root_direct": array(output.decoder.joints_root_direct),
        "ray_field": array(output.ray_field),
    }
    for key in ("wrist_prior", "wrist_log_scale", "reliability_logits", "local_roi_valid", "local_roi_bounds", "anchor_quality"):
        value = getattr(output.decoder, key, None)
        if value is not None:
            payload[key] = array(value)
    for key in ("geometry_weight", "joint_weights"):
        value = getattr(output.pnp, key, None)
        if value is not None:
            payload[key] = array(value)
    if batch.get("timestamps") is not None:
        payload["timestamps"] = batch["timestamps"][0].detach().cpu().double().numpy()
    if output.camera_fit is not None:
        payload.update(
            {
                "camera_fit_focal": array(output.camera_fit.focal),
                "camera_fit_principal": array(output.camera_fit.principal),
                "camera_fit_bearing_variance": array(output.camera_fit.bearing_variance),
                "camera_fit_valid": array(output.camera_fit.valid),
                "camera_fit_rms_normalized": array(output.camera_fit.rms_normalized),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def build_system(
    root: Path, config: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> tuple[HandPrismSystem, WanFrozenVAEEncoder]:
    model_dir = resolve(root, config["model_dir"])
    videox_fun = resolve(root, config["videox_fun"])
    vae = load_official_vae(model_dir / "Wan2.2_VAE.pth", videox_fun, torch_dtype=dtype).to(device)
    vae_encoder = WanFrozenVAEEncoder(vae)
    wan = load_official_wan(model_dir, videox_fun, torch_dtype=dtype)
    report = inject_wan_lora(wan)
    if report.modules != 300 or report.parameters != 161_218_560:
        raise RuntimeError("official LoRA schema changed")
    configure_trainable_backbone(wan)
    encoder = WanCleanLatentEncoder(wan, gradient_checkpointing=False)
    mano = SmplxMano(resolve(root, config["mano_model"]), flat_hand_mean=True)
    return (
        HandPrismSystem(
            encoder,
            mano,
            architecture=config["architecture"],
            decoder_config=decoder_config_from_json(config),
            solver_config=solver_config_from_json(config),
            fusion_config=fusion_config_from_json(config),
        )
        .to(device)
        .eval(),
        vae_encoder,
    )


@torch.no_grad()
def canonical_joints(system: HandPrismSystem, device: torch.device) -> torch.Tensor:
    identity = torch.eye(3, device=device).view(1, 1, 1, 3, 3)
    global_rotation = identity.expand(1, 1, 2, 3, 3)
    articulation = identity.unsqueeze(-3).expand(1, 1, 2, 15, 3, 3)
    betas = torch.zeros(1, 1, 2, 10, device=device)
    joints, _ = system.hand.mano(global_rotation, articulation, betas)
    return joints[0, 0]


def main() -> int:
    parser = argparse.ArgumentParser()
    add_architecture_argument(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("arctic", "hot3d"),
        help="optional subset; defaults to every dataset in dataset_weights",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-segments-per-dataset", type=int)
    parser.add_argument("--no-dump", action="store_true")
    weights_mode = parser.add_mutually_exclusive_group()
    weights_mode.add_argument(
        "--legacy-weights", action="store_true",
        help="Core inference only: accept one of the two SHA-256-pinned preserved final weights",
    )
    weights_mode.add_argument("--released-weights", action="store_true",
                             help="load SHA-256-pinned Core safetensors from the GitHub release")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(resolve(root, str(args.config)), architecture=args.architecture)
    configured_datasets = tuple(config["dataset_weights"])
    datasets = tuple(args.datasets) if args.datasets is not None else configured_datasets
    if len(set(datasets)) != len(datasets):
        raise ValueError("evaluation datasets must be unique")
    outside_config = set(datasets) - set(configured_datasets)
    if outside_config:
        raise ValueError(
            f"evaluation datasets are not enabled by the config: {sorted(outside_config)}"
        )
    checkpoint_path = allowed_path(resolve(root, str(args.checkpoint)))
    output_dir = v3_run_path(root, args.output)
    checkpoint_digest = sha256(checkpoint_path)
    if args.released_weights:
        checkpoint = load_released_weights(checkpoint_path, config, architecture=args.architecture)
    else:
        if args.legacy_weights:
            require_legacy_digest(args.architecture, config["solver"], checkpoint_digest)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        validate_checkpoint_identity(
            checkpoint, config, architecture=args.architecture,
            legacy_sha256=checkpoint_digest if args.legacy_weights else None,
        )
    rank, world, device = distributed_context()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    dtype = torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float16
    system, vae_encoder = build_system(root, config, device, dtype)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None and checkpoint_config.get("solver") != config["solver"]:
        raise RuntimeError("checkpoint solver and evaluation config differ")
    if checkpoint_config is not None and checkpoint_config.get("dataset_weights") != config.get(
        "dataset_weights"
    ):
        raise RuntimeError("checkpoint and evaluation dataset mixtures differ")
    load_trainable_state(system, checkpoint["trainable"])
    checkpoint_step = int(checkpoint["step"])
    canonical = canonical_joints(system, device)
    manifest_root = resolve(root, config["manifests"])
    mano_path = resolve(root, config["mano_model"])
    summaries: dict[str, dict[str, float | int | None]] = {}
    overall = EvaluationAccumulator()
    for dataset_name in datasets:
        dataset = HandPrismWindowDataset(
            manifest_root / f"{dataset_name}_test.jsonl",
            mano_model_path=mano_path,
            training=False,
            **dataset_options(config, training=False),
        )
        sampler = DistributedEvalSampler(len(dataset), rank, world)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=collate_samples,
            **loader_options(args.workers),
        )
        accumulator = EvaluationAccumulator()
        progress_path = output_dir / f"progress_rank_{rank:03d}.jsonl"
        with progress_path.open("a", encoding="utf-8") as progress:
            for local_index, host_batch in enumerate(loader):
                if (
                    args.max_segments_per_dataset is not None
                    and local_index >= args.max_segments_per_dataset
                ):
                    break
                started = time.perf_counter()
                batch = move_batch(host_batch, device)
                video = batch["video"].to(dtype)
                with torch.inference_mode():
                    latent = vae_encoder(video)
                    with torch.autocast("cuda", dtype=dtype):
                        output = system(
                            latent,
                            target_frames=video.shape[2],
                            solver=config["solver"],
                            intrinsics=batch["intrinsics"],
                            image_size=batch["image_size"],
                            distortion=batch["distortion"],
                            calibration_ray_field=(
                                batch["gt_ray_field"] if config["solver"] == "standard" else None
                            ),
                            camera_model=batch["camera_model"],
                            camera_parameters=batch["camera_parameters"],
                            source_image_size=batch["source_image_size"],
                            **fusion_forward_options(config, batch),
                        )
                    from handprism.training import scoring_batch
                    batch = scoring_batch(batch, *output.ray_field.shape[1:3])
                    _, gt_vertices_root = system.hand.mano(
                        batch["global_rotation"],
                        batch["articulation"],
                        batch["betas"],
                    )
                    score_batch(
                        accumulator,
                        output,
                        batch,
                        gt_vertices_root,
                        canonical,
                        solver=config["solver"],
                        gt_mano_translation=batch["translation"]
                        - system.hand.mano.root_offset(batch["betas"]),
                    )
                frame_start = int(batch["frame_indices"][0, 0])
                if not args.no_dump:
                    name = prediction_name(dataset_name, batch["recording_id"][0], frame_start)
                    save_prediction(
                        output_dir / "predictions" / dataset_name / name,
                        output,
                        batch,
                        checkpoint_step,
                        config["solver"],
                        args.architecture,
                    )
                row = {
                    "dataset": dataset_name,
                    "recording_id": batch["recording_id"][0],
                    "frame_start": frame_start,
                    "rank": rank,
                    "seconds": time.perf_counter() - started,
                }
                progress.write(json.dumps(row, sort_keys=True) + "\n")
                progress.flush()
                if rank == 0 and (local_index + 1) % 10 == 0:
                    print(json.dumps({"type": "evaluation_progress", **row}), flush=True)
                del video, latent, output, gt_vertices_root, batch
        state = accumulator.as_tensor(device)
        if world > 1:
            dist.all_reduce(state, op=dist.ReduceOp.SUM)
        accumulator.load_tensor(state)
        result = accumulator.finalize()
        validate_metric_result(result)
        if rank == 0:
            summaries[dataset_name] = result
            for key, value in accumulator.values.items():
                overall.values[key] += value
            print(
                json.dumps(
                    {"type": "evaluation_dataset", "dataset": dataset_name, **result},
                    sort_keys=True,
                ),
                flush=True,
            )
    if rank == 0:
        overall_result = overall.finalize()
        validate_metric_result(overall_result)
        report = {
            "format": "handprism-dataset-mixture-evaluation-v2-clean",
            "metric_protocol_version": 2,
            "method": architecture_spec(args.architecture).display_name,
            "architecture": args.architecture,
            "implementation_id": architecture_spec(args.architecture).implementation_id,
            "legacy_weights": args.legacy_weights,
            "released_weights": args.released_weights,
            "pnp_residual_kind": (
                "bearing_diagonal_proxy" if args.architecture == "handprism-core" else "native_pixels"
            ),
            "translation_metric_contract": "CT=MANO-native-translation; Wrist=J0+translation",
            "full_test": args.max_segments_per_dataset is None
            and set(datasets) == {"arctic", "hot3d"},
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_step": checkpoint_step,
            "solver": config["solver"],
            "world_size": world,
            "datasets": summaries,
            "overall": overall_result,
        }
        if config["solver"] == "kfree":
            report["kfree_camera_fit"] = config["kfree_camera_fit"]
        require_finite_json(report)
        temporary_metrics = output_dir / "metrics.json.tmp"
        temporary_metrics.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary_metrics.replace(output_dir / "metrics.json")
        print(json.dumps({"type": "evaluation_complete", **overall_result}, sort_keys=True))
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
