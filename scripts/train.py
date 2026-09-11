#!/usr/bin/env python3
"""Distributed, resumable HandPrism training on ARCTIC and HOT3D."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

import argparse
from contextlib import nullcontext
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from handprism.backbone import (
    WanCleanLatentEncoder,
    WanFrozenVAEEncoder,
    load_official_vae,
    load_official_wan,
)
from handprism.config import DecoderConfig, OptimizerConfig, SolverConfig
from handprism.fusion_runtime import (fusion_config_from_json, loss_weights_from_json,
    dataset_options, fusion_forward_options, validation_selection_score, weighted_loss_audit, ddp_options,
    training_diagnostics, decoder_gradient_norms)
from handprism.data import HandPrismWindowDataset, collate_samples
from handprism.data.dataset import read_jsonl
from handprism.data.difficulty import stratified_order, validation_rank_limit, validate_fusion_index
from handprism.data.policy import SUPPORTED_DATASETS, allowed_path, manifest_path, validate_record
from handprism.data.schema import MANIFEST_SCHEMA, supports_manifest_schema
from handprism.architectures import (
    CORE, FUSION, CONTRACT_VERSION, CHECKPOINT_FORMAT, add_architecture_argument,
    architecture_spec, require_config_architecture, validate_checkpoint_identity,
)
from handprism.evaluator import EvaluationAccumulator, score_batch, validate_metric_result
from handprism.lora import (
    configure_trainable_backbone,
    inject_wan_lora,
    promote_trainable_parameters,
)
from handprism.losses import HandPrismLoss, camera_fit_supervision
from handprism.mano import SmplxMano
from handprism.paths import v3_run_path
from handprism.system import HandPrismSystem
from handprism.training import (
    build_optimizer,
    prediction_from_output,
    scoring_batch,
    target_from_batch,
    trainable_parameter_report,
    warmup_cosine,
)


WAN_REVISION = "b8bc1a65ab71d054ba4636dc0dac104aa4df2686"
WAN_CONFIG_SHA256 = "dc20f8568e6b08121aa8e388c8cddba2ed42e9e5e57c7d75fbb6bf3b771cd018"
WAN_DIT_SHA256 = "ace4718a7c87ee3e5606a68ab79142c4395e81aece76b8120bc886f0fbbe1d16"
WAN_VAE_SHA256 = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
VIDEOX_FUN_COMMIT = "6f3fb60dad9b6a60ff6f962e62cffa11cafb084b"
VIDEOX_FUN_TREE_SHA256 = "4e9184033cf1ff2a2aa0b2362c2e72ffb42498cdcb7b29547db401e2169ed647"
LOSS_NAMES = (
    "total",
    "rotation_geodesic",
    "rotation_matrix",
    "shape",
    "joints_root",
    "joints_camera",
    "wrist",
    "anchors_2d",
    "reprojection_joints",
    "reprojection_wrist",
    "translation",
    "existence",
    "visibility",
    "acceleration",
    "ray",
    "camera_fit",
    "joints_root_mano", "direct_mano_consistency", "wrist_prior", "reliability",
    "log_depth",
    "velocity_error", "acceleration_error", "wrist_velocity_error", "wrist_acceleration_error",
)
def distributed_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return rank, world, local_rank, device


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)


def load_config(path: Path, *, architecture: str | None = None) -> dict[str, Any]:
    value = json.loads(allowed_path(path).read_text())
    return validate_config(value, architecture=architecture)


def validate_config(value: dict[str, Any], *, architecture: str | None = None) -> dict[str, Any]:
    """Validate before writing generated configurations or touching run state."""
    required = {
        "seed",
        "solver",
        "steps",
        "batch_size_per_gpu",
        "gradient_accumulation",
        "dataset_weights",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"training config is missing {sorted(missing)}")
    for key in ("steps", "batch_size_per_gpu", "gradient_accumulation", "validate_every", "checkpoint_every"):
        if key in value and (type(value[key]) is not int or value[key] <= 0):
            raise ValueError(f"{key} must be a positive integer")
    weights = value["dataset_weights"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError("dataset_weights must be a non-empty object")
    unsupported = set(weights) - set(SUPPORTED_DATASETS)
    if unsupported:
        raise ValueError(f"unsupported datasets: {sorted(unsupported)}")
    if any(
        not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) <= 0.0
        for weight in weights.values()
    ):
        raise ValueError("dataset weights must be finite and positive")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
        raise ValueError("dataset weights must sum to one")
    if value["solver"] not in {"standard", "kfree"}:
        raise ValueError("solver must be standard or kfree")
    fit = value.get("kfree_camera_fit")
    if value["solver"] == "kfree":
        if not isinstance(fit, dict) or fit.get("enabled") is not True:
            raise ValueError("K-free requires enabled kfree_camera_fit")
        required_fit = {
            "variance_floor",
            "focal_min",
            "focal_max",
            "loss_weight",
            "warmup_steps",
            "loss_norm",
            "assumption_note",
        }
        missing_fit = required_fit - fit.keys()
        if missing_fit:
            raise ValueError(f"kfree_camera_fit is missing {sorted(missing_fit)}")
        numeric = (
            float(fit["variance_floor"]),
            float(fit["focal_min"]),
            float(fit["focal_max"]),
            float(fit["loss_weight"]),
        )
        if not all(math.isfinite(item) for item in numeric):
            raise ValueError("kfree_camera_fit numeric values must be finite")
        if not math.isclose(numeric[0], 1e-4, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("HandPrism requires camera-fit variance_floor=1e-4")
        if not 0.0 < numeric[1] < numeric[2]:
            raise ValueError("kfree_camera_fit focal bracket must be positive and ordered")
        if not math.isclose(numeric[3], 5.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("HandPrism requires camera-fit loss_weight=5")
        if int(fit["warmup_steps"]) != 500:
            raise ValueError("HandPrism requires a 500-step L_fit warmup")
        if fit["loss_norm"] not in {"l1", "smooth_l1"}:
            raise ValueError("kfree_camera_fit loss_norm must be l1 or smooth_l1")
        if not str(fit["assumption_note"]).strip():
            raise ValueError("camera-fit settings require a non-empty assumption_note")
    elif fit is not None:
        raise ValueError("standard solver must not configure the K-free camera fit")
    selected = architecture if architecture is not None else value.get("architecture")
    require_config_architecture(value, selected)
    if value.get("loss_reduction") != "per_clip":
        raise ValueError("HandPrism requires explicit per_clip loss_reduction")
    if value.get("geometry_dtype") != "float32":
        raise ValueError("HandPrism requires explicit float32 geometry_dtype")
    if set(value.get("decoder", {})) != {"anchor_offset_cells"}:
        raise ValueError("decoder must explicitly declare anchor_offset_cells")
    decoder_config_from_json(value)
    fusion_config_from_json(value)
    loss_weights_from_json(value)
    solver_config_from_json(value)
    if selected == FUSION:
        if not isinstance(value.get("fusion"), dict):
            raise ValueError("Fusion requires explicit fusion settings")
        policy = value.get("validation_selection", "legacy_mean_loss")
        if policy not in {"legacy_mean_loss", "accuracy_coverage_v2"}:
            raise ValueError("unknown validation selection policy")
        if policy == "accuracy_coverage_v2":
            limit = value.get("validation_clips_per_dataset")
            if type(limit) is not int or limit <= 0 or "validation_batches_per_dataset" in value:
                raise ValueError("Fusion requires a positive global validation_clips_per_dataset, not a per-rank budget")
            interval = value.get("full_validate_every", 0)
            if type(interval) is not int or interval <= 0 or interval % int(value["validate_every"]) or interval % int(value["checkpoint_every"]):
                raise ValueError("full validation must coincide with validation/checkpoint boundaries")
        fraction = value.get("hard_window_fraction", 0.)
        if type(fraction) not in (int, float) or not 0 <= fraction <= .5:
            raise ValueError("hard_window_fraction must be in [0,.5]")
        dataset_options(value, training=True)
    elif value.get("geometry_refinement") or value.get("augmentation") or value.get("hard_window_fraction"):
        raise ValueError("Core data/solver settings must remain unchanged")
    if selected == CORE and value["decoder"]["anchor_offset_cells"] != 0:
        raise ValueError("HandPrism-Core does not support anchor offsets")
    if weights != {"arctic": 0.4375, "hot3d": 0.5625}:
        raise ValueError("HandPrism requires ARCTIC/HOT3D weights 0.4375/0.5625")
    roots = value.get("dataset_roots", {})
    if set(roots) != set(SUPPORTED_DATASETS):
        raise ValueError("dataset_roots must contain exactly ARCTIC/HOT3D")
    for path_value in (*roots.values(), value["manifests"]):
        allowed_path(path_value)
    if fit is not None:
        if not 0 < float(fit.get("max_rms_normalized", 0)) <= 0.1:
            raise ValueError("explicit camera fit residual limit must be in (0,0.1]")
        targets = {"core_bearings"} if selected == CORE else {
            "pinhole_compatible", "effective_camera", "raw_bearings"
        }
        if fit.get("target") not in targets:
            raise ValueError("kfree_camera_fit must declare a supported target")
    return value


def solver_config_from_json(config: dict[str, Any]) -> SolverConfig:
    """Construct the selected architecture's geometric solver and fit guards."""

    fit = config.get("kfree_camera_fit", {})
    geometry = config.get("geometry_refinement", {})
    if set(geometry) - {"robust_iterations", "robust_huber_px", "depth_refine_fraction"}:
        raise ValueError("unknown geometry refinement setting")
    return SolverConfig(
        architecture=config.get("architecture", FUSION),
        camera_fit_variance_floor=float(fit.get("variance_floor", 1e-4)),
        camera_fit_focal_min=float(fit.get("focal_min", 0.05)),
        camera_fit_focal_max=float(fit.get("focal_max", 10.0)),
        camera_fit_max_rms_normalized=float(fit.get("max_rms_normalized", 0.01)),
        camera_fit_target=str(fit.get("target", "pinhole_compatible")),
        **geometry,
    )


def decoder_config_from_json(config: dict[str, Any]) -> DecoderConfig:
    return DecoderConfig(
        anchor_offset_cells=float(config.get("decoder", {}).get("anchor_offset_cells", 0.0))
    )


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def loader_options(workers: int) -> dict[str, Any]:
    output: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        output["prefetch_factor"] = 1
    return output


class DeterministicDrawBatchSampler(Sampler[list[tuple[int, int]]]):
    """Infinite, rank-disjoint sample stream recoverable from an occurrence index."""

    def __init__(
        self,
        length: int,
        batch_size: int,
        rank: int,
        world: int,
        seed: int,
        namespace: str,
    ) -> None:
        if length <= 0 or batch_size <= 0:
            raise ValueError("dataset length and batch size must be positive")
        self.length = length
        self.batch_size = batch_size
        self.rank = rank
        self.world = world
        self.seed = seed
        self.namespace = namespace
        self.start_batch = 0
        self._permutations: dict[int, Tensor] = {}

    def set_start_batch(self, value: int) -> None:
        if value < 0:
            raise ValueError("start batch cannot be negative")
        self.start_batch = value

    def _permutation(self, epoch: int) -> Tensor:
        if epoch not in self._permutations:
            digest = hashlib.sha256(
                f"{self.seed}:{self.namespace}:permutation:{epoch}".encode()
            ).digest()
            generator = torch.Generator().manual_seed(
                int.from_bytes(digest[:8], "big") % (2**63 - 1)
            )
            self._permutations = {epoch: torch.randperm(self.length, generator=generator)}
        return self._permutations[epoch]

    def __iter__(self):
        batch_index = self.start_batch
        global_batch = self.world * self.batch_size
        while True:
            batch: list[tuple[int, int]] = []
            for slot in range(self.batch_size):
                global_sample = batch_index * global_batch + self.rank * self.batch_size + slot
                epoch, offset = divmod(global_sample, self.length)
                record_index = int(self._permutation(epoch)[offset])
                digest = hashlib.sha256(
                    f"{self.seed}:{self.namespace}:window:{global_sample}".encode()
                ).digest()
                draw_seed = int.from_bytes(digest[:8], "big")
                batch.append((record_index, draw_seed))
            batch_index += 1
            yield batch

    def __len__(self) -> int:
        return 2**31


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without DistributedSampler's duplicate padding rows."""

    def __init__(self, length: int, rank: int, world: int) -> None:
        self.length = length
        self.rank = rank
        self.world = world

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world))

    def __len__(self) -> int:
        if self.rank >= self.length:
            return 0
        return (self.length - 1 - self.rank) // self.world + 1


def make_loaders(
    root: Path,
    config: dict[str, Any],
    rank: int,
    world: int,
) -> tuple[
    dict[str, DataLoader],
    dict[str, DeterministicDrawBatchSampler],
    dict[str, DataLoader],
    dict[str, DistributedEvalSampler],
]:
    manifest_root = resolve(root, config["manifests"])
    if config.get("validation_selection") == "accuracy_coverage_v2" or config.get("hard_window_fraction", 0):
        audit = manifest_report(manifest_root, tuple(config["dataset_weights"]))
        records = {name: {split: read_jsonl(manifest_root / f"{name}_{split}.jsonl")
                         for split in ("train", "val")} for name in config["dataset_weights"]}
        validate_fusion_index(audit["split_report"], records, config.get("validation_clips_per_dataset", 0))
    mano = resolve(root, config["mano_model"])
    workers = int(config.get("num_workers", 2))
    train_loaders, train_samplers = {}, {}
    val_loaders, val_samplers = {}, {}
    for name in config["dataset_weights"]:
        train_dataset = HandPrismWindowDataset(
            manifest_root / f"{name}_train.jsonl",
            mano_model_path=mano,
            training=True,
            **dataset_options(config, training=True),
        )
        val_dataset = HandPrismWindowDataset(
            manifest_root / f"{name}_val.jsonl",
            mano_model_path=mano,
            training=False,
            **dataset_options(config, training=False),
        )
        train_sampler = DeterministicDrawBatchSampler(
            len(train_dataset),
            int(config["batch_size_per_gpu"]),
            rank,
            world,
            int(config["seed"]),
            name,
        )
        if config.get("validation_selection") == "accuracy_coverage_v2":
            val_dataset.records = [val_dataset.records[i] for i in stratified_order(val_dataset.records)]
        val_sampler = DistributedEvalSampler(len(val_dataset), rank, world)
        train_loaders[name] = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_samples,
            **loader_options(workers),
        )
        val_loaders[name] = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=val_sampler,
            drop_last=False,
            collate_fn=collate_samples,
            **loader_options(max(0, workers // 2)),
        )
        train_samplers[name] = train_sampler
        val_samplers[name] = val_sampler
    return train_loaders, train_samplers, val_loaders, val_samplers


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def dataset_for_step(config: dict[str, Any], step_index: int) -> str:
    names = tuple(config["dataset_weights"])
    weights = tuple(float(config["dataset_weights"][name]) for name in names)
    digest = hashlib.sha256(f"{config['seed']}:dataset-step:{step_index}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name, weight in zip(names, weights):
        cumulative += weight
        if value < cumulative:
            return name
    return names[-1]


class LoaderCycle:
    def __init__(self, loaders: dict[str, DataLoader]):
        self.loaders = loaders
        self.iterators = {name: iter(loader) for name, loader in loaders.items()}

    def next(self, name: str) -> dict[str, Any]:
        return next(self.iterators[name])


def consumed_dataset_batches(config: dict[str, Any], completed_steps: int) -> dict[str, int]:
    counts = {name: 0 for name in config["dataset_weights"]}
    accumulation = int(config["gradient_accumulation"])
    for step_index in range(completed_steps):
        counts[dataset_for_step(config, step_index)] += accumulation
    return counts


def reduce_mean(value: Tensor, world: int) -> Tensor:
    result = value.detach().float().clone()
    if world > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= world
    return result


def trainable_state(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(module: nn.Module, state: dict[str, Tensor]) -> None:
    parameters = dict(module.named_parameters())
    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.requires_grad and name not in state
    ]
    unexpected = [
        name for name in state if name not in parameters or not parameters[name].requires_grad
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint trainable schema mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    wrong_shapes = [name for name in state if state[name].shape != parameters[name].shape]
    if wrong_shapes:
        raise RuntimeError(f"checkpoint tensor shapes differ: {wrong_shapes[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))


def save_checkpoint(
    run_dir: Path,
    step: int,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_validation: float,
    world: int,
    config: dict[str, Any],
    rng_by_rank: list[dict[str, Any]],
) -> Path:
    require_config_architecture(config, config["architecture"])
    if getattr(module, "architecture", None) != config["architecture"]:
        raise ValueError("cannot save a model under another architecture's configuration")
    directory = run_dir / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step:06d}.pt"
    temporary = directory / f".step_{step:06d}.tmp"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "contract_version": CONTRACT_VERSION,
            "architecture": config["architecture"],
            "implementation_id": architecture_spec(config["architecture"]).implementation_id,
            "step": step,
            "best_validation": best_validation,
            "world_size": world,
            "config": config,
            "trainable": trainable_state(module),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_by_rank": rng_by_rank,
        },
        temporary,
    )
    temporary.replace(path)
    (directory / "latest.json").write_text(
        json.dumps({
            "step": step, "path": path.name, "architecture": config["architecture"],
            "implementation_id": config["implementation_id"],
        }, sort_keys=True) + "\n"
    )
    return path


def preflight_resume(path: Path, config: dict[str, Any], world: int) -> None:
    """Reject mismatched resume metadata on CPU before CUDA/data initialization."""
    require_config_architecture(config, config.get("architecture"))
    checkpoint = torch.load(allowed_path(path), map_location="cpu", weights_only=False, mmap=True)
    validate_checkpoint_identity(checkpoint, config, architecture=config["architecture"])
    if checkpoint.get("world_size") != world:
        raise RuntimeError("exact resume requires the checkpoint's original world size")


def restore_checkpoint(
    path: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rank: int,
    world: int,
    expected_config: dict[str, Any],
) -> tuple[int, float]:
    require_config_architecture(expected_config, expected_config.get("architecture"))
    checkpoint = torch.load(allowed_path(path), map_location="cpu", weights_only=False)
    validate_checkpoint_identity(
        checkpoint, expected_config, architecture=expected_config["architecture"]
    )
    if getattr(module, "architecture", None) != expected_config["architecture"]:
        raise ValueError("model architecture differs from resume configuration")
    if int(checkpoint.get("world_size", world)) != world:
        raise RuntimeError("exact resume requires the checkpoint's original world size")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None and checkpoint_config != expected_config:
        raise RuntimeError("checkpoint config differs from the requested run config")
    load_trainable_state(module, checkpoint["trainable"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    rng_by_rank = checkpoint.get("rng_by_rank")
    if rng_by_rank is not None:
        if len(rng_by_rank) != world:
            raise RuntimeError("checkpoint RNG state does not match world size")
        rng = rng_by_rank[rank]
        torch.set_rng_state(rng["torch"])
        if isinstance(rng["cuda"], list):
            torch.cuda.set_rng_state_all(rng["cuda"])
        else:
            torch.cuda.set_rng_state(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    else:
        # Backward compatibility for diagnostic checkpoints created before the
        # rank-specific RNG contract was introduced.
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
    return int(checkpoint["step"]), float(checkpoint.get("best_validation", float("inf")))


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    """Hash file paths and contents while ignoring VCS/runtime metadata."""

    digest = hashlib.sha256()
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and ".git" not in path.relative_to(root).parts
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix != ".pyc"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{sha256(path)}  ./{relative}\n".encode())
    return digest.hexdigest()


def git_fingerprint(root: Path) -> dict[str, Any]:
    repository = git_value(root, "rev-parse", "--show-toplevel")
    if not repository or Path(repository).resolve() != root.resolve():
        # A source snapshot inside an ignored run directory is not the parent
        # checkout. Its identity is the source/config hashes in the contract.
        return {
            "commit": None, "dirty": None, "dirty_diff_sha256": None,
            "untracked_files": [], "scope": "unversioned-source-snapshot",
        }
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    untracked = sorted(
        path.decode("utf-8", errors="surrogateescape")
        for path in untracked_raw.split(b"\0")
        if path
    )
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(diff)
    files: list[dict[str, str]] = []
    for relative in untracked:
        path = root / relative
        if not path.is_file():
            continue
        value = sha256(path)
        files.append({"path": relative, "sha256": value})
        digest.update(b"untracked\0")
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(value.encode())
    return {
        "scope": "workspace-repository",
        "commit": git_value(root, "rev-parse", "HEAD"),
        "dirty": bool(diff or files),
        "dirty_diff_sha256": digest.hexdigest(),
        "untracked_files": files,
    }


def package_versions() -> dict[str, str | None]:
    packages = (
        "av",
        "diffusers",
        "h5py",
        "numpy",
        "projectaria-tools",
        "safetensors",
        "scipy",
        "smplx",
        "torch",
        "transformers",
    )
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def manifest_report(manifest_root: Path, datasets: tuple[str, ...]) -> dict[str, Any]:
    allowed_path(manifest_root)
    if set(datasets) != set(SUPPORTED_DATASETS):
        raise ValueError("only the ARCTIC/HOT3D manifest set is allowed")
    split_path = manifest_root / "split_report.json"
    report = json.loads(split_path.read_text())
    if not supports_manifest_schema(report.get("version")):
        raise RuntimeError("training requires the v2_clean data contract")
    expected_datasets = set(datasets)
    reported_datasets = set(report.get("datasets", {}))
    if reported_datasets != expected_datasets:
        raise RuntimeError(
            "split report dataset mismatch: "
            f"configured={sorted(expected_datasets)} reported={sorted(reported_datasets)}"
        )
    selected = report.get("selected_datasets")
    if selected is not None and set(selected) != expected_datasets:
        raise RuntimeError("split report selected_datasets does not match configuration")
    expected = {
        f"{dataset}_{split}.jsonl" for dataset in datasets for split in ("train", "val", "test")
    }
    actual = {path.name for path in manifest_root.glob("*.jsonl")}
    if actual != expected:
        raise RuntimeError(f"manifest set mismatch: {actual ^ expected}")
    manifests: dict[str, Any] = {}
    identities: dict[str, dict[str, set[str]]] = {}
    split_groups: dict[str, dict[str, set[str]]] = {}
    records: dict[str, dict[str, list[dict]]] = {name: {} for name in datasets}
    for path in sorted(manifest_root.glob("*.jsonl")):
        manifest_path(path)
        dataset, split = path.stem.rsplit("_", 1)
        rows = 0
        recording_ids: set[str] = set()
        group_ids: set[str] = set()
        records[dataset][split] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                validate_record(record, path)
                records[dataset][split].append(record)
                rows += 1
                recording_ids.add(str(record["recording_id"]))
                group_ids.add(str(record["split_group"]))
        digest = sha256(path)
        declared = report["datasets"][dataset][split]
        if (
            rows != int(declared["rows"])
            or len(recording_ids) != int(declared["recordings"])
            or len(group_ids) != int(declared["split_groups"])
            or digest != declared["sha256"]
        ):
            raise RuntimeError(f"manifest does not match split_report.json: {path}")
        manifests[path.name] = {
            "sha256": digest,
            "rows": rows,
            "recordings": len(recording_ids),
            "split_groups": len(group_ids),
        }
        identities.setdefault(dataset, {})[split] = recording_ids
        split_groups.setdefault(dataset, {})[split] = group_ids
    for dataset, splits in identities.items():
        if (
            splits["train"] & splits["val"]
            or splits["train"] & splits["test"]
            or splits["val"] & splits["test"]
        ):
            raise RuntimeError(f"recording leakage detected in {dataset}")
    for dataset, splits in split_groups.items():
        if (
            splits["train"] & splits["val"]
            or splits["train"] & splits["test"]
            or splits["val"] & splits["test"]
        ):
            raise RuntimeError(f"split-group leakage detected in {dataset}")
    # Normalize only the in-memory report; retain the digest of the original
    # file for provenance and never rewrite source manifests during an audit.
    source_schema_sha256 = hashlib.sha256(report["version"].encode()).hexdigest()
    report["version"] = MANIFEST_SCHEMA
    if "fusion_index" in report:
        validate_fusion_index(report, records)
    return {
        "split_report_sha256": sha256(split_path),
        "source_schema_sha256": source_schema_sha256,
        "split_report": report,
        "manifests": manifests,
    }


def local_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def gather_rng_states(rank: int, world: int) -> list[dict[str, Any]] | None:
    local = local_rng_state()
    if world == 1:
        return [local]
    gathered: list[dict[str, Any] | None] | None = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank != 0:
        return None
    assert gathered is not None and all(item is not None for item in gathered)
    return [item for item in gathered if item is not None]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def implementation_notes(config: dict[str, Any]) -> list[str]:
    """Describe this run's choices without making benchmark equivalence claims."""
    decoder = decoder_config_from_json(config)
    notes = [
        "Only ARCTIC and HOT3D are enabled; manifest hashes and split counts are recorded separately",
        "Frozen manifests use recording- and subject/participant-disjoint splits",
        f"Decoder uses {decoder.heads} attention heads and FFN width {decoder.ffn_dim}",
        "Wan features use the official 32x backbone spatial compression",
    ]
    if config["architecture"] == FUSION:
        notes.extend([
            "Fusion modules, native RGB detail, motion objectives and sampling are explicit ablations",
            "Timestamp-aware motion uses valid intervals only; unknown occlusion is not a negative label",
            "Only full validation may select accuracy_coverage_v2 best checkpoints; test never selects",
        ])
    if config["solver"] == "kfree":
        fit = config["kfree_camera_fit"]
        notes.append(
            f"K-free fit uses normalized focal bounds [{fit['focal_min']}, "
            f"{fit['focal_max']}], {fit['loss_norm']} loss, target {fit['target']}; "
            f"{fit['assumption_note']}"
        )
    return notes


def write_contract(
    root: Path,
    run_dir: Path,
    config: dict[str, Any],
    world: int,
    system: HandPrismSystem,
    optimizer: torch.optim.Optimizer,
    *,
    allow_existing: bool,
) -> None:
    manifest_root = resolve(root, config["manifests"])
    model_dir = resolve(root, config["model_dir"])
    mano_dir = resolve(root, config["mano_model"])
    asset_hashes = {
        "wan_config": sha256(model_dir / "config.json"),
        "wan_dit": sha256(model_dir / "diffusion_pytorch_model.safetensors"),
        "wan_vae": sha256(model_dir / "Wan2.2_VAE.pth"),
        "mano_left": sha256(mano_dir / "MANO_LEFT.pkl"),
        "mano_right": sha256(mano_dir / "MANO_RIGHT.pkl"),
    }
    if asset_hashes["wan_dit"] != WAN_DIT_SHA256:
        raise RuntimeError("Wan DiT SHA-256 differs from the audited revision")
    if asset_hashes["wan_vae"] != WAN_VAE_SHA256:
        raise RuntimeError("Wan VAE SHA-256 differs from the audited revision")
    if asset_hashes["wan_config"] != WAN_CONFIG_SHA256:
        raise RuntimeError("Wan config SHA-256 differs from the audited revision")
    videox_root = resolve(root, config["videox_fun"])
    videox_tree = source_tree_sha256(videox_root)
    if videox_tree != VIDEOX_FUN_TREE_SHA256:
        raise RuntimeError("VideoX-Fun source tree differs from the audited commit")
    parameters = trainable_parameter_report(
        system.encoder.model,
        system.hand.decoder,
        system.hand.ray_head,
    )
    parameters["system_total"] = sum(value.numel() for value in system.parameters())
    parameters["system_requires_grad"] = sum(
        value.numel() for value in system.parameters() if value.requires_grad
    )
    parameters["optimizer_groups"] = {
        str(group.get("name", index)): sum(value.numel() for value in group["params"])
        for index, group in enumerate(optimizer.param_groups)
    }
    parameters["optimizer_group_dtypes"] = {
        str(group.get("name", index)): sorted({str(value.dtype) for value in group["params"]})
        for index, group in enumerate(optimizer.param_groups)
    }
    datasets = tuple(config["dataset_weights"])
    stable = {
        "implementation_contract_version": CONTRACT_VERSION,
        "implementation_id": architecture_spec(config["architecture"]).implementation_id,
        "architecture": config["architecture"],
        "method": architecture_spec(config["architecture"]).display_name,
        "geometry_dtype": "float32",
        "loss_reduction": "per_clip",
        "data_geometry_contract": ("fusion-r4-safe-targets-log-depth-discrete-roi"
                                   if config["architecture"] == FUSION else
                                   "hot3d-pca-and-canonical-left-shapedirs-consistent-r2"),
        "metric_protocol_version": 2,
        "metric_translation": "CT=MANO-native-translation; Wrist=J0+translation",
        # The isolated server copy is inside an ignored runs/ directory. Its
        # parent's clean Git status alone does not identify its actual code.
        "source_sha256": {
            "src": source_tree_sha256(root / "src"),
            "scripts": source_tree_sha256(root / "scripts"),
        },
        "git": git_fingerprint(root),
        "config": config,
        "world_size": world,
        "effective_batch": int(config["batch_size_per_gpu"])
        * int(config["gradient_accumulation"])
        * world,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "packages": package_versions(),
            "gpus": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        },
        "wan": {
            "huggingface_repo": "alibaba-pai/Wan2.2-Fun-5B-Control",
            "huggingface_url": "https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control",
            "downloaded_from": "https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-Control",
            "revision": WAN_REVISION,
            "videox_fun_repo": "https://github.com/aigc-apps/VideoX-Fun",
            "videox_fun_commit": VIDEOX_FUN_COMMIT,
            "videox_fun_tree_sha256": videox_tree,
        },
        "asset_sha256": asset_hashes,
        "data": manifest_report(manifest_root, datasets),
        "parameters": parameters,
        "implementation_notes": implementation_notes(config),
    }
    stable_payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    contract = {
        "format": "handprism-run-contract-v2",
        "created_unix": time.time(),
        "hostname": socket.gethostname(),
        "command": sys.argv,
        "resume_guard_sha256": hashlib.sha256(stable_payload.encode()).hexdigest(),
        **stable,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "contract.json"
    if path.exists():
        if not allow_existing:
            raise FileExistsError(
                f"run directory already has a contract; use --resume auto: {run_dir}"
            )
        existing = json.loads(path.read_text())
        if existing.get("resume_guard_sha256") != contract["resume_guard_sha256"]:
            raise RuntimeError("resume contract mismatch; refusing to mix two experiments")
        with (run_dir / "resume_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "time_unix": time.time(),
                        "command": sys.argv,
                        "architecture": config["architecture"],
                        "implementation_id": architecture_spec(config["architecture"]).implementation_id,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return
    atomic_json(path, contract)


@torch.no_grad()
def validate(
    system: nn.Module,
    vae_encoder: WanFrozenVAEEncoder,
    criterion: HandPrismLoss,
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    device: torch.device,
    world: int,
    dtype: torch.dtype,
) -> dict[str, float]:
    system.eval()
    result: dict[str, Any] = {}
    core = system.module if isinstance(system, DistributedDataParallel) else system
    identity = torch.eye(3, device=device).view(1, 1, 1, 3, 3)
    canonical, _ = core.hand.mano(
        identity.expand(1, 1, 2, 3, 3),
        identity.unsqueeze(-3).expand(1, 1, 2, 15, 3, 3),
        torch.zeros(1, 1, 2, 10, device=device),
    )
    maximum = int(config.get("validation_batches_per_dataset", 8))
    if config.get("validation_selection") == "accuracy_coverage_v2":
        maximum = validation_rank_limit(config["validation_clips_per_dataset"],
                                        dist.get_rank() if world > 1 else 0, world)
    elif maximum == 0:
        maximum = None
    for name, loader in loaders.items():
        loss_sums = torch.zeros(len(LOSS_NAMES), device=device)
        summary = torch.zeros(4, device=device)
        geometry = torch.zeros(18, device=device, dtype=torch.float64)
        fit_activity = torch.zeros(3, device=device)
        accumulator = EvaluationAccumulator()
        extrema = torch.tensor([float("inf"), float("-inf"), 0.0], device=device)
        nonfinite = torch.zeros(4, device=device)
        from itertools import islice
        for host_batch in islice(loader, maximum):
            batch = move_batch(host_batch, device)
            video = batch["video"].to(dtype)
            latent = vae_encoder(video)
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
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
                prediction = prediction_from_output(output)
                target = target_from_batch(
                    batch, output.ray_field.shape[1], output.ray_field.shape[2]
                )
                losses = criterion(
                    prediction,
                    target,
                    batch["intrinsics"],
                    batch["image_size"],
                    batch["distortion"],
                    camera_model=batch["camera_model"],
                    camera_parameters=batch["camera_parameters"],
                    source_image_size=batch["source_image_size"],
                    solver=config["solver"],
                    optimizer_step=int(config["steps"]),
                    camera_fit_warmup_steps=int(
                        config.get("kfree_camera_fit", {}).get("warmup_steps", 500)
                    ),
                    camera_fit_loss_norm=str(
                        config.get("kfree_camera_fit", {}).get("loss_norm", "l1")
                    ),
                    camera_fit_config=solver_config_from_json(config),
                )
            batch = scoring_batch(batch, *output.ray_field.shape[1:3])
            distance = (prediction.joints_root_direct.float() - batch["joints_root"]).norm(dim=-1)
            mask = batch["valid_joints_3d"].float()
            mano_mask = mask * batch["valid_mano"].float().unsqueeze(-1)
            mano_distance = (output.joints_root_mano.float() - batch["joints_root"]).norm(dim=-1)
            pixel_scale = batch["image_size"][:, None, None, [1, 0]]
            anchor_error = (
                (output.decoder.anchors_2d.float() - batch["joints_2d"]) * pixel_scale.unsqueeze(-2)
            ).norm(dim=-1)
            mask_2d = batch["valid_joints_2d"].float()
            hands = batch["valid_hand"].float()
            edge_mask = mask_2d * (
                ((batch["joints_2d"] < 0.05) | (batch["joints_2d"] > 0.95)).any(-1)
            )
            wrist_error = (output.pnp.translation - batch["translation"]).norm(dim=-1)
            depth_error = (output.pnp.translation[..., 2] - batch["translation"][..., 2]).abs()
            wrist_mask = batch["valid_mano"].float() * hands
            extent = output.decoder.anchors_2d.amax(-2) - output.decoder.anchors_2d.amin(-2)
            box_diag = (extent.float() * pixel_scale).norm(dim=-1)
            geometry += torch.stack(
                (
                    (distance * mask).sum(),
                    mask.sum(),
                    (mano_distance * mano_mask).sum(),
                    mano_mask.sum(),
                    (anchor_error * mask_2d).sum(),
                    mask_2d.sum(),
                    (output.pnp.used_fallback * hands).sum(),
                    hands.sum(),
                    ((~output.pnp.projection_valid) * hands).sum(),
                    ((box_diag < 5.0) * hands).sum(),
                    ((output.pnp.vote_count < core.hand.solver_config.min_votes) * hands).sum(),
                    ((output.pnp.rms_pixels > output.pnp.residual_limit_pixels) * hands).sum(),
                    (anchor_error * edge_mask).sum(),
                    edge_mask.sum(),
                    (wrist_error * wrist_mask).sum(),
                    wrist_mask.sum(),
                    (depth_error * wrist_mask).sum(),
                    wrist_mask.sum(),
                )
            ).double()
            _, gt_vertices = core.hand.mano(
                batch["global_rotation"], batch["articulation"], batch["betas"]
            )
            score_batch(
                accumulator,
                output,
                batch,
                gt_vertices,
                canonical[0, 0],
                solver=config["solver"],
                gt_mano_translation=batch["translation"]
                - core.hand.mano.root_offset(batch["betas"]),
            )
            for term_index, term in enumerate(LOSS_NAMES):
                loss_sums[term_index] += losses[term].float()
            summary[0] += (distance * mask).sum() / mask.sum().clamp_min(1)
            summary[1] += output.pnp.used_fallback.float().mean()
            summary[2] += 1
            if output.camera_fit is not None:
                summary[3] += output.camera_fit.valid.float().mean()
                _, _, activity = camera_fit_supervision(
                    output.camera_fit,
                    target.ray_field,
                    target.valid_ray,
                    solver_config_from_json(config),
                )
                fit_activity += torch.stack(list(activity.values()))
            depth = output.decoder.depth.float()
            translation = output.pnp.translation.float()
            extrema[0] = torch.minimum(extrema[0], depth.amin())
            extrema[1] = torch.maximum(extrema[1], depth.amax())
            extrema[2] = torch.maximum(extrema[2], translation.abs().amax())
            nonfinite += torch.stack(
                (
                    (~torch.isfinite(depth)).sum(),
                    (~torch.isfinite(translation)).sum(),
                    (~torch.isfinite(output.joints_camera)).sum(),
                    (~torch.isfinite(output.ray_field)).sum(),
                )
            )
        if world > 1:
            dist.all_reduce(geometry, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(summary, op=dist.ReduceOp.SUM)
            dist.all_reduce(extrema[:1], op=dist.ReduceOp.MIN)
            dist.all_reduce(extrema[1:], op=dist.ReduceOp.MAX)
            dist.all_reduce(nonfinite, op=dist.ReduceOp.SUM)
            dist.all_reduce(fit_activity, op=dist.ReduceOp.SUM)
        metric_state = accumulator.as_tensor(device)
        if world > 1:
            dist.all_reduce(metric_state, op=dist.ReduceOp.SUM)
        accumulator.load_tensor(metric_state)
        dataset_metrics = accumulator.finalize()
        validate_metric_result(dataset_metrics)
        for key, value in dataset_metrics.items():
            result[f"val/{name}/test_protocol/{key}"] = value
        count = summary[2].clamp_min(1)
        result[f"val/{name}/windows"] = int(summary[2].cpu())
        for term_index, term in enumerate(LOSS_NAMES):
            result[f"val/{name}/{term}"] = float((loss_sums[term_index] / count).cpu())
        result[f"val/{name}/direct_root_mpjpe_mm"] = float(
            (1000 * geometry[0] / geometry[1].clamp_min(1)).cpu()
        )
        result[f"val/{name}/mano_root_mpjpe_mm"] = float(
            (1000 * geometry[2] / geometry[3].clamp_min(1)).cpu()
        )
        result[f"val/{name}/anchors_epe_px"] = float((geometry[4] / geometry[5].clamp_min(1)).cpu())
        result[f"val/{name}/edge_anchors_count"] = int(geometry[13].cpu())
        result[f"val/{name}/edge_anchors_epe_px"] = (
            float((geometry[12] / geometry[13]).cpu()) if geometry[13] > 0 else None
        )
        result[f"val/{name}/wrist_error_mm"] = float(
            (1000 * geometry[14] / geometry[15].clamp_min(1)).cpu()
        )
        result[f"val/{name}/depth_error_mm"] = float(
            (1000 * geometry[16] / geometry[17].clamp_min(1)).cpu()
        )
        for key, numerator in (
            ("valid_hand_fallback_fraction", 6),
            ("projection_failure_fraction", 8),
            ("anchor_box_under_5px_fraction", 9),
            ("insufficient_votes_fraction", 10),
            ("large_pixel_residual_fraction", 11),
        ):
            result[f"val/{name}/{key}"] = float(
                (geometry[numerator] / geometry[7].clamp_min(1)).cpu()
            )
        result[f"val/{name}/fallback_fraction"] = float((summary[1] / count).cpu())
        if config["solver"] == "kfree":
            result[f"val/{name}/camera_fit_valid_fraction"] = float((summary[3] / count).cpu())
            for index, key in enumerate(
                (
                    "predicted_numerical_valid_fraction",
                    "target_compatible_fraction",
                    "supervised_ray_fraction",
                )
            ):
                result[f"val/{name}/camera_fit_{key}"] = float((fit_activity[index] / count).cpu())
        result[f"val/{name}/depth_min_m"] = float(extrema[0].cpu())
        result[f"val/{name}/depth_max_m"] = float(extrema[1].cpu())
        result[f"val/{name}/translation_abs_max_m"] = float(extrema[2].cpu())
        for key, value in zip(("depth", "translation", "joints_camera", "ray_field"), nonfinite):
            result[f"val/{name}/nonfinite_{key}"] = int(value.cpu())
    result["val/mean_loss"] = sum(
        value for key, value in result.items() if key.endswith("/total")
    ) / len(loaders)
    system.train()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    add_architecture_argument(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="stop cleanly at this step without changing the configured scheduler horizon",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(resolve(root, str(args.config)), architecture=args.architecture)
    if args.steps is not None:
        if "ablation" in config:
            raise ValueError("regenerate ablation clip budget instead of overriding --steps")
        config["steps"] = args.steps
        validate_config(config, architecture=args.architecture)
    if "ablation" in config and config["ablation"]["world_size"] != int(os.environ.get("WORLD_SIZE", "1")):
        raise ValueError("world size disagrees with the frozen ablation exposure budget")
    identity = {"architecture": args.architecture, "implementation_id": config["implementation_id"]}
    run_dir = v3_run_path(root, args.run_dir)
    resume_path: Path | None = None
    if args.resume == "auto":
        latest = allowed_path(run_dir / "checkpoints/latest.json")
        if latest.is_file():
            resume_path = latest.parent / json.loads(latest.read_text())["path"]
    elif args.resume:
        resume_path = resolve(root, args.resume)

    if resume_path is not None:
        preflight_resume(resume_path, config, int(os.environ.get("WORLD_SIZE", "1")))
    rank, world, _local_rank, device = distributed_context()
    seed_everything(int(config["seed"]), rank)
    dtype = torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float16
    train_loaders, train_samplers, val_loaders, _ = make_loaders(root, config, rank, world)

    model_dir = resolve(root, config["model_dir"])
    videox_fun = resolve(root, config["videox_fun"])
    vae = load_official_vae(model_dir / "Wan2.2_VAE.pth", videox_fun, torch_dtype=dtype).to(device)
    vae_encoder = WanFrozenVAEEncoder(vae)
    wan = load_official_wan(model_dir, videox_fun, torch_dtype=dtype)
    lora_report = inject_wan_lora(wan)
    if lora_report.modules != 300 or lora_report.parameters != 161_218_560:
        raise RuntimeError("official LoRA schema changed")
    configure_trainable_backbone(wan)
    if config.get("trainable_dtype", "float32") != "float32":
        raise ValueError("the audited mixed-precision path requires FP32 trainable parameters")
    promote_trainable_parameters(wan)
    encoder = WanCleanLatentEncoder(
        wan, gradient_checkpointing=bool(config.get("gradient_checkpointing", True))
    )
    mano = SmplxMano(resolve(root, config["mano_model"]), flat_hand_mean=True)
    system = HandPrismSystem(
        encoder,
        mano,
        architecture=args.architecture,
        decoder_config=decoder_config_from_json(config),
        solver_config=solver_config_from_json(config),
        fusion_config=fusion_config_from_json(config),
    ).to(device)
    criterion = HandPrismLoss(loss_weights_from_json(config), fusion_config_from_json(config)).to(device)
    optimizer_config = OptimizerConfig(
        steps=int(config["steps"]),
        warmup_steps=int(config.get("warmup_steps", 200)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        gradient_clip=float(config.get("gradient_clip", 1.0)),
        decoder_lr=float(config["learning_rates"]["decoder"]),
        lora_lr=float(config["learning_rates"]["lora"]),
        patch_lr=float(config["learning_rates"]["patch_embedding"]),
        clips_per_gpu=int(config["batch_size_per_gpu"]),
    )
    optimizer = build_optimizer(wan, system.hand.decoder, system.hand.ray_head, optimizer_config)
    scheduler = warmup_cosine(optimizer, optimizer_config)
    if rank == 0:
        write_contract(
            root,
            run_dir,
            config,
            world,
            system,
            optimizer,
            allow_existing=bool(args.resume or args.validate_only),
        )
        atomic_json(
            run_dir / "run_state.json",
            {
                **identity,
                "status": "initializing",
                "time_unix": time.time(),
                "resume_checkpoint": str(resume_path) if resume_path else None,
            },
        )
    if world > 1:
        dist.barrier()
    start_step, best_validation = 0, float("inf")
    if resume_path is not None:
        start_step, best_validation = restore_checkpoint(
            resume_path, system, optimizer, scheduler, rank, world, config
        )

    if world > 1:
        system_wrapped: nn.Module = DistributedDataParallel(
            system,
            device_ids=[device.index],
            output_device=device.index,
            **ddp_options(config),
        )
    else:
        system_wrapped = system
    if args.validate_only:
        metrics = validate(
            system_wrapped,
            vae_encoder,
            criterion,
            val_loaders,
            config,
            device,
            world,
            dtype,
        )
        if rank == 0:
            row = {
                **identity,
                "type": "validation_only",
                "checkpoint": str(resume_path) if resume_path is not None else None,
                "time_unix": time.time(),
                **metrics,
            }
            (run_dir / "validation_only.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n"
            )
            atomic_json(
                run_dir / "run_state.json",
                {**identity, "status": "validation_complete", "time_unix": time.time()},
            )
            print(json.dumps(row, sort_keys=True), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0
    consumed = consumed_dataset_batches(config, start_step)
    for name, sampler in train_samplers.items():
        sampler.set_start_batch(consumed[name])
    cycle = LoaderCycle(train_loaders)
    optimizer.zero_grad(set_to_none=True)
    log_path = run_dir / "train.jsonl"
    log_handle = log_path.open("a", encoding="utf-8") if rank == 0 else None
    accumulation = int(config["gradient_accumulation"])
    configured_final_step = int(config["steps"])
    final_step = (
        min(configured_final_step, args.stop_after_step)
        if args.stop_after_step is not None
        else configured_final_step
    )
    if final_step < start_step:
        raise ValueError("--stop-after-step is earlier than the resumed checkpoint")
    if rank == 0:
        atomic_json(
            run_dir / "run_state.json",
            {
                **identity,
                "status": "running",
                "time_unix": time.time(),
                "start_step": start_step,
                "target_step": configured_final_step,
                "this_invocation_stop": final_step,
            },
        )
    try:
        for step in range(start_step + 1, final_step + 1):
            selection_improved = False
            started = time.perf_counter()
            term_sums: dict[str, Tensor] = {}
            diagnostic_sums: dict[str, Tensor] = {}
            camera_fit_valid_sum = torch.zeros((), device=device)
            camera_fit_activity_sum = torch.zeros(3, device=device)
            dataset_counts = {name: 0 for name in config["dataset_weights"]}
            # One dataset is selected for the whole global optimizer step.
            # Gradient accumulation shards that batch for 40GB GPUs; every
            # microbatch and rank therefore uses the same dataset.
            dataset_name = dataset_for_step(config, step - 1)
            dataset_counts[dataset_name] = accumulation
            for micro in range(accumulation):
                batch = move_batch(cycle.next(dataset_name), device)
                video = batch["video"].to(dtype)
                latent = vae_encoder(video)
                synchronize = micro == accumulation - 1
                context = (
                    nullcontext()
                    if synchronize or not isinstance(system_wrapped, DistributedDataParallel)
                    else system_wrapped.no_sync()
                )
                with context, torch.autocast("cuda", dtype=dtype):
                    output = system_wrapped(
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
                        **fusion_forward_options(config, batch, training=True, step=step),
                    )
                    prediction = prediction_from_output(output)
                    target = target_from_batch(
                        batch, output.ray_field.shape[1], output.ray_field.shape[2]
                    )
                    losses = criterion(
                        prediction,
                        target,
                        batch["intrinsics"],
                        batch["image_size"],
                        batch["distortion"],
                        camera_model=batch["camera_model"],
                        camera_parameters=batch["camera_parameters"],
                        source_image_size=batch["source_image_size"],
                        solver=config["solver"],
                        optimizer_step=step,
                        camera_fit_warmup_steps=int(
                            config.get("kfree_camera_fit", {}).get("warmup_steps", 500)
                        ),
                        camera_fit_loss_norm=str(
                            config.get("kfree_camera_fit", {}).get("loss_norm", "l1")
                        ),
                        camera_fit_config=solver_config_from_json(config),
                    )
                    (losses["total"] / accumulation).backward()
                    if output.camera_fit is not None:
                        camera_fit_valid_sum += (
                            output.camera_fit.valid.float().mean().detach() / accumulation
                        )
                        with torch.no_grad():
                            _, _, activity = camera_fit_supervision(
                                output.camera_fit,
                                target.ray_field,
                                target.valid_ray,
                                solver_config_from_json(config),
                            )
                            camera_fit_activity_sum += (
                                torch.stack(list(activity.values())) / accumulation
                            )
                for name, value in losses.items():
                    term_sums[name] = (
                        term_sums.get(name, torch.zeros_like(value.detach()))
                        + value.detach() / accumulation
                    )
                if config["architecture"] == FUSION:
                    for name, value in training_diagnostics(output, batch).items():
                        diagnostic_sums[name] = diagnostic_sums.get(name, torch.zeros_like(value)) + value / accumulation
                del video, latent, output, prediction, target, losses
            diagnostic_values = {}
            if config["architecture"] == FUSION and step % int(config.get("log_every", 1)) == 0:
                diagnostic_sums.update(decoder_gradient_norms(system.hand.decoder))
                reduced = reduce_mean(torch.stack(list(diagnostic_sums.values())), world)
                if rank == 0:
                    diagnostic_values = dict(zip(diagnostic_sums, reduced.cpu().tolist()))
            trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(config.get("gradient_clip", 1.0)), error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0 and step % int(config.get("log_every", 1)) == 0:
                row: dict[str, Any] = {
                    **identity,
                    "type": "train",
                    "step": step,
                    "time_unix": time.time(),
                    "seconds": time.perf_counter() - started,
                    "grad_norm": float(reduce_mean(grad_norm, world).cpu()),
                    "max_cuda_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "datasets": dataset_counts,
                    "global_clips_seen": step * world * int(config["batch_size_per_gpu"]) * accumulation,
                    "global_frames_seen": step * world * int(config["batch_size_per_gpu"]) * accumulation * 81,
                    "lr": {
                        group.get("name", str(index)): group["lr"]
                        for index, group in enumerate(optimizer.param_groups)
                    },
                }
                if config["solver"] == "kfree":
                    fit = config["kfree_camera_fit"]
                    row["camera_fit_effective_weight"] = float(fit["loss_weight"]) * min(
                        float(step) / float(fit["warmup_steps"]), 1.0
                    )
                    row["camera_fit_valid_fraction"] = float(
                        reduce_mean(camera_fit_valid_sum, world).cpu()
                    )
                    reduced_activity = reduce_mean(camera_fit_activity_sum, world).cpu().tolist()
                    for key, value in zip(
                        (
                            "predicted_numerical_valid_fraction",
                            "target_compatible_fraction",
                            "supervised_ray_fraction",
                        ),
                        reduced_activity,
                    ):
                        row[f"camera_fit_{key}"] = float(value)
                    row["camera_fit_target"] = fit["target"]
                row.update(
                    {
                        f"loss/{name}": float(reduce_mean(value, world).cpu())
                        for name, value in term_sums.items()
                    }
                )
                assert log_handle is not None
                if config["architecture"] == FUSION:
                    row["fusion_diagnostics"] = diagnostic_values
                    row["loss_audit"] = weighted_loss_audit(
                        {name: row[f"loss/{name}"] for name in term_sums}, config, step)
                log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                log_handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                atomic_json(
                    run_dir / "run_state.json",
                    {
                        **identity,
                        "status": "running",
                        "time_unix": time.time(),
                        "step": step,
                        "target_step": configured_final_step,
                        "seconds_last_step": row["seconds"],
                    },
                )
            elif world > 1:
                # Every rank must participate in the reductions performed by rank zero.
                if step % int(config.get("log_every", 1)) == 0:
                    reduce_mean(grad_norm, world)
                    if config["solver"] == "kfree":
                        reduce_mean(camera_fit_valid_sum, world)
                        reduce_mean(camera_fit_activity_sum, world)
                    for value in term_sums.values():
                        reduce_mean(value, world)

            if step % int(config.get("validate_every", 500)) == 0 or step == final_step:
                modern_selection = config.get("validation_selection") == "accuracy_coverage_v2"
                full_validation = modern_selection and (
                    step % int(config["full_validate_every"]) == 0 or step == configured_final_step)
                validation_config = dict(config)
                if full_validation:
                    validation_config["validation_clips_per_dataset"] = 0
                metrics = validate(
                    system_wrapped,
                    vae_encoder,
                    criterion,
                    val_loaders,
                    validation_config,
                    device,
                    world,
                    dtype,
                )
                if rank == 0:
                    metrics["val/full_validation"] = full_validation
                    selection_score = (validation_selection_score(metrics) if full_validation
                                       else metrics["val/mean_loss"] if not modern_selection else None)
                    metrics["val/selection_score"] = selection_score
                    selection_improved = selection_score is not None and selection_score < best_validation
                    if selection_improved:
                        best_validation = selection_score
                    row = {**identity, "type": "validation", "step": step, "time_unix": time.time(), **metrics}
                    assert log_handle is not None
                    log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    log_handle.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)
                    if step >= 500:
                        for dataset in SUPPORTED_DATASETS:
                            issues = [
                                key
                                for key, limit in (
                                    ("valid_hand_fallback_fraction", 0.95),
                                    ("anchor_box_under_5px_fraction", 0.8),
                                )
                                if metrics[f"val/{dataset}/{key}"] >= limit
                            ]
                            if (
                                config["solver"] == "kfree"
                                and metrics[f"val/{dataset}/camera_fit_supervised_ray_fraction"]
                                == 0
                            ):
                                issues.append("no_camera_fit_supervision")
                            if issues:
                                warning = {
                                    **identity,
                                    "type": "quality_warning",
                                    "step": step,
                                    "time_unix": time.time(),
                                    "dataset": dataset,
                                    "issues": issues,
                                    "note": "Engineering diagnostic; assess accuracy on held-out validation data",
                                }
                                log_handle.write(json.dumps(warning, sort_keys=True) + "\n")
                                log_handle.flush()
                                print(json.dumps(warning, sort_keys=True), flush=True)

            if step % int(config.get("checkpoint_every", 500)) == 0 or step == final_step:
                rng_by_rank = gather_rng_states(rank, world)
                if rank == 0:
                    assert rng_by_rank is not None
                    path = save_checkpoint(
                        run_dir,
                        step,
                        system,
                        optimizer,
                        scheduler,
                        best_validation,
                        world,
                        config,
                        rng_by_rank,
                    )
                    if selection_improved:
                        atomic_json(run_dir / "checkpoints" / "best.json", {
                            **identity, "step": step, "path": path.name,
                            "score": best_validation,
                            "selection": config.get("validation_selection", "legacy_mean_loss"),
                            "metric_protocol_version": 2,
                            "full_validation": full_validation,
                        })
                    print(
                        json.dumps({**identity, "type": "checkpoint", "step": step, "path": str(path)}),
                        flush=True,
                    )
                if world > 1:
                    dist.barrier()
        if rank == 0:
            atomic_json(
                run_dir / "run_state.json",
                {
                    **identity,
                    "status": "complete"
                    if final_step == configured_final_step
                    else "stopped_cleanly",
                    "time_unix": time.time(),
                    "step": final_step,
                    "target_step": configured_final_step,
                },
            )
    except BaseException as error:
        if rank == 0:
            atomic_json(
                run_dir / "run_state.json",
                {
                    **identity,
                    "status": "failed",
                    "time_unix": time.time(),
                    "step": locals().get("step", start_step),
                    "target_step": configured_final_step,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise
    finally:
        if log_handle is not None:
            log_handle.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
