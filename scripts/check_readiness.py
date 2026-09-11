#!/usr/bin/env python3
"""Fail closed before launching a configured HandPrism ARCTIC/HOT3D run."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

import argparse
import hashlib
from importlib import metadata, util
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any


WAN_REVISION = "b8bc1a65ab71d054ba4636dc0dac104aa4df2686"
WAN_CONFIG_SHA256 = "dc20f8568e6b08121aa8e388c8cddba2ed42e9e5e57c7d75fbb6bf3b771cd018"
WAN_DIT_SHA256 = "ace4718a7c87ee3e5606a68ab79142c4395e81aece76b8120bc886f0fbbe1d16"
WAN_VAE_SHA256 = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
VIDEOX_FUN_COMMIT = "6f3fb60dad9b6a60ff6f962e62cffa11cafb084b"
VIDEOX_FUN_TREE_SHA256 = "4e9184033cf1ff2a2aa0b2362c2e72ffb42498cdcb7b29547db401e2169ed647"
MANO_SHA256 = {
    "LEFT": "c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30",
    "RIGHT": "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767",
}
from handprism.data.policy import SUPPORTED_DATASETS, allowed_path, manifest_path, validate_record
from handprism.data.schema import supports_manifest_schema
from handprism.architectures import CORE, add_architecture_argument, require_config_architecture

SPLITS = ("train", "val", "test")
EXPECTED_WEIGHT_POLICIES = {
    ("arctic", "hot3d"): {"arctic": 0.4375, "hot3d": 0.5625},
}
EXPECTED_CAPABILITIES = {
    "arctic": {
        "mano",
        "joints_root_3d",
        "camera_3d",
        "exact_2d",
        "ray",
        "existence",
        "visibility",
    },
    "hot3d": {
        "mano",
        "joints_root_3d",
        "camera_3d",
        "exact_2d",
        "ray",
        "existence",
        "visibility",
    },
}
HOT3D_REQUIRED_MASKS = {
    "mask_hand_pose_available",
    "mask_headset_pose_available",
    "mask_good_exposure",
    "mask_qa_pass",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
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


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def add_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks[name] = {"pass": bool(passed), "detail": detail}


def required_paths(record: dict[str, Any]) -> list[Path]:
    dataset = record["dataset"]
    if dataset == "arctic":
        root = Path(record["root"])
        subject, sequence = str(record["sequence"]).split("/", 1)
        base = root / "data" / "raw_seqs" / subject
        return [
            base / f"{sequence}.mano.npy",
            base / f"{sequence}.egocam.dist.npy",
            root / "data" / "images" / subject / sequence / "0",
        ]
    if dataset == "hot3d":
        root = Path(record["recording_root"])
        return [
            root / "recording.vrs",
            root / "mano_hand_pose_trajectory.jsonl",
            root / "headset_trajectory.csv",
            root / "masks" / "mask_hand_pose_available.csv",
            root / "masks" / "mask_headset_pose_available.csv",
            root / "masks" / "mask_good_exposure.csv",
            root / "masks" / "mask_qa_pass.csv",
            root / "masks" / "mask_hand_visible.csv",
        ]
    raise ValueError(f"unknown dataset {dataset!r}")


def audit_manifests(
    manifest_root: Path,
    datasets: tuple[str, ...],
    checks: dict[str, dict[str, Any]],
    check_data_files: bool,
) -> dict[str, Any]:
    allowed_path(manifest_root)
    if set(datasets) - set(SUPPORTED_DATASETS):
        raise ValueError("unsupported dataset in audit")
    split_path = manifest_root / "split_report.json"
    if not split_path.is_file():
        add_check(checks, "split_report", False, str(split_path))
        return {}
    report = json.loads(split_path.read_text())
    add_check(
        checks,
        "split_report_schema",
        supports_manifest_schema(report.get("version")) and int(report.get("frames_per_window", 0)) == 81,
        {"path": str(split_path), "sha256": sha256(split_path)},
    )
    summary: dict[str, Any] = {}
    expected_files = {f"{dataset}_{split}.jsonl" for dataset in datasets for split in SPLITS}
    actual_files = {path.name for path in manifest_root.glob("*.jsonl")}
    report_datasets = set(report.get("datasets", {}))
    selected = report.get("selected_datasets")
    manifest_set_pass = (
        actual_files == expected_files
        and report_datasets == set(datasets)
        and (selected is None or set(selected) == set(datasets))
    )
    add_check(
        checks,
        "manifest_dataset_set",
        manifest_set_pass,
        {
            "configured": list(datasets),
            "files": sorted(actual_files),
            "reported": sorted(report_datasets),
            "selected_datasets": selected,
        },
    )
    identity_sets: dict[str, dict[str, set[str]]] = {name: {} for name in datasets}
    group_sets: dict[str, dict[str, set[str]]] = {name: {} for name in datasets}
    missing_paths: list[str] = []
    checked_paths: set[str] = set()
    for dataset in datasets:
        for split in SPLITS:
            path = manifest_root / f"{dataset}_{split}.jsonl"
            rows = 0
            identities: set[str] = set()
            groups: set[str] = set()
            schema_errors: list[str] = []
            manifest_path(path)
            if path.is_file():
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        validate_record(record, path)
                        rows += 1
                        required = {
                            "dataset",
                            "recording_id",
                            "split_group",
                            "num_frames",
                            "frames",
                            "split",
                            "capabilities",
                        }
                        if required - record.keys():
                            schema_errors.append(
                                f"line {line_number}: missing {sorted(required - record.keys())}"
                            )
                            continue
                        if record["dataset"] != dataset or record["split"] != split:
                            schema_errors.append(f"line {line_number}: dataset/split mismatch")
                        if int(record["frames"]) != 81 or int(record["num_frames"]) < 81:
                            schema_errors.append(f"line {line_number}: invalid frame count")
                        if set(record.get("capabilities", ())) != EXPECTED_CAPABILITIES[dataset]:
                            schema_errors.append(
                                f"line {line_number}: capability contract mismatch"
                            )
                        has_bounds = {"start_min", "start_max"} <= record.keys()
                        has_ranges = "valid_ranges" in record
                        if split == "train" and not (has_bounds or has_ranges):
                            schema_errors.append(
                                f"line {line_number}: missing random-window domain"
                            )
                        if split != "train" and "start_frame" not in record:
                            schema_errors.append(f"line {line_number}: missing fixed start_frame")
                        if split == "train" and has_bounds:
                            lower, upper = int(record["start_min"]), int(record["start_max"])
                            if lower < 0 or upper < lower or upper + 81 > int(record["num_frames"]):
                                schema_errors.append(f"line {line_number}: invalid train bounds")
                        if split == "train" and has_ranges:
                            previous_stop = 0
                            for start, stop in record["valid_ranges"]:
                                start, stop = int(start), int(stop)
                                if (
                                    start < previous_stop
                                    or stop - start < 81
                                    or stop > int(record["num_frames"])
                                ):
                                    schema_errors.append(
                                        f"line {line_number}: invalid/non-disjoint valid_ranges"
                                    )
                                    break
                                previous_stop = stop
                        if split != "train" and "start_frame" in record:
                            start = int(record["start_frame"])
                            if start < 0 or start + 81 > int(record["num_frames"]):
                                schema_errors.append(f"line {line_number}: invalid fixed window")
                        if dataset == "arctic" and int(record.get("image_offset", -1)) < 0:
                            schema_errors.append(f"line {line_number}: missing ARCTIC ioi_offset")
                        if (
                            dataset == "hot3d"
                            and set(record.get("required_masks", ())) != HOT3D_REQUIRED_MASKS
                        ):
                            schema_errors.append(
                                f"line {line_number}: HOT3D mask contract mismatch"
                            )
                        identities.add(str(record["recording_id"]))
                        groups.add(str(record["split_group"]))
                        if check_data_files:
                            for source in required_paths(record):
                                allowed_path(source)
                                source_text = str(source)
                                if source_text in checked_paths:
                                    continue
                                checked_paths.add(source_text)
                                if not source.exists() and len(missing_paths) < 20:
                                    missing_paths.append(source_text)
            digest = sha256(path) if path.is_file() else None
            declared = report.get("datasets", {}).get(dataset, {}).get(split, {})
            passed = (
                path.is_file()
                and not schema_errors
                and rows == int(declared.get("rows", -1))
                and len(identities) == int(declared.get("recordings", -1))
                and len(groups) == int(declared.get("split_groups", -1))
                and digest == declared.get("sha256")
            )
            key = f"manifest_{dataset}_{split}"
            detail = {
                "path": str(path),
                "rows": rows,
                "recordings": len(identities),
                "split_groups": len(groups),
                "sha256": digest,
                "schema_errors": schema_errors[:10],
            }
            add_check(checks, key, passed, detail)
            summary[key] = detail
            identity_sets[dataset][split] = identities
            group_sets[dataset][split] = groups
    overlap: dict[str, Any] = {}
    for dataset, splits in identity_sets.items():
        values = {
            "train_val": len(splits["train"] & splits["val"]),
            "train_test": len(splits["train"] & splits["test"]),
            "val_test": len(splits["val"] & splits["test"]),
        }
        overlap[dataset] = values
        add_check(checks, f"zero_recording_overlap_{dataset}", not any(values.values()), values)
        group_values = {
            "train_val": len(group_sets[dataset]["train"] & group_sets[dataset]["val"]),
            "train_test": len(group_sets[dataset]["train"] & group_sets[dataset]["test"]),
            "val_test": len(group_sets[dataset]["val"] & group_sets[dataset]["test"]),
        }
        add_check(
            checks,
            f"zero_split_group_overlap_{dataset}",
            not any(group_values.values()),
            group_values,
        )
    if check_data_files:
        add_check(
            checks,
            "all_manifest_source_paths_exist",
            not missing_paths,
            {"unique_paths_checked": len(checked_paths), "missing_first_20": missing_paths},
        )
    summary["recording_overlap"] = overlap
    return summary


def audit_config(
    root: Path,
    path: Path,
    manifest_root: Path,
    datasets: tuple[str, ...],
    checks: dict[str, dict[str, Any]],
    *, architecture: str | None = None,
) -> None:
    config = json.loads(allowed_path(path).read_text())
    selected = architecture if architecture is not None else config.get("architecture")
    try:
        from scripts.train import load_config
        load_config(path, architecture=selected)
        require_config_architecture(config, selected)
        architecture_pass = True
    except ValueError:
        architecture_pass = False
    solver = config.get("solver")
    expected_batch = 64 if solver == "standard" else 32
    effective_batch = (
        int(config.get("batch_size_per_gpu", 0)) * int(config.get("gradient_accumulation", 0)) * 8
    )
    expected_weights = EXPECTED_WEIGHT_POLICIES.get(datasets)
    weights = config.get("dataset_weights", {})
    weight_policy_matches = (
        expected_weights is not None
        and set(weights) == set(expected_weights)
        and all(
            math.isclose(float(weights[name]), float(expected_weights[name]), abs_tol=1e-10)
            for name in expected_weights
        )
    )
    configured_manifest = Path(config.get("manifests", ""))
    if not configured_manifest.is_absolute():
        configured_manifest = root / configured_manifest
    allowed_path(configured_manifest)
    dataset_roots = config.get("dataset_roots", {})
    fit = config.get("kfree_camera_fit")
    if selected != CORE and (config.get("hard_window_fraction", 0)
                             or config.get("validation_selection") == "accuracy_coverage_v2"):
        try:
            from handprism.data.difficulty import validate_fusion_index
            from handprism.data.dataset import read_jsonl
            index_report = json.loads((configured_manifest / "split_report.json").read_text())
            records = {name: {split: read_jsonl(configured_manifest / f"{name}_{split}.jsonl")
                             for split in ("train", "val")} for name in datasets}
            validate_fusion_index(index_report, records, config.get("validation_clips_per_dataset", 0))
            index_ok = True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            index_ok = False
        add_check(checks, f"fusion_difficulty_index_{solver}", index_ok,
                  "Requires index v2, nonoverlapping multiwindow full val larger than fast val; no test geometry")
    fit_config_pass = solver == "standard" and fit is None
    if solver == "kfree" and isinstance(fit, dict):
        try:
            fit_config_pass = (
                fit.get("enabled") is True
                and math.isclose(float(fit.get("variance_floor")), 1e-4, abs_tol=1e-12)
                and math.isclose(float(fit.get("focal_min")), 0.05, abs_tol=1e-12)
                and math.isclose(float(fit.get("focal_max")), 10.0, abs_tol=1e-12)
                and math.isclose(float(fit.get("loss_weight")), 5.0, abs_tol=1e-12)
                and int(fit.get("warmup_steps")) == 500
                and fit.get("loss_norm") == "l1"
                and fit.get("target") in (
                    {"core_bearings"} if selected == CORE else
                    {"pinhole_compatible", "effective_camera", "raw_bearings"}
                )
                and math.isclose(float(fit.get("max_rms_normalized")), 0.01, abs_tol=1e-10)
                and bool(str(fit.get("assumption_note", "")).strip())
            )
        except (TypeError, ValueError):
            fit_config_pass = False
    passed = (
        architecture_pass
        and solver in {"standard", "kfree"}
        and int(config.get("steps", 0)) == 20_000
        and int(config.get("warmup_steps", 0)) == 200
        and config.get("dtype") == "bfloat16"
        and config.get("trainable_dtype") == "float32"
        and config.get("geometry_dtype") == "float32"
        and config.get("loss_reduction") == "per_clip"
        and set(config.get("decoder", {})) == {"anchor_offset_cells"}
        and 0 <= float(config["decoder"]["anchor_offset_cells"]) <= 1
        and (selected != CORE or float(config["decoder"]["anchor_offset_cells"]) == 0)
        and tuple(weights) == datasets
        and weight_policy_matches
        and set(dataset_roots) == set(datasets)
        and effective_batch == expected_batch
        and configured_manifest.resolve() == manifest_root.resolve()
        and configured_manifest.is_dir()
        and fit_config_pass
    )
    add_check(
        checks,
        f"config_{solver}",
        passed,
        {
            "path": str(path),
            "architecture": selected,
            "sha256": sha256(path),
            "steps": config.get("steps"),
            "datasets": list(weights),
            "weights": weights,
            "expected_weights": expected_weights,
            "manifest_root": str(configured_manifest),
            "effective_batch_on_8_gpus": effective_batch,
            "expected_effective_batch": expected_batch,
            "kfree_camera_fit": fit,
            "kfree_camera_fit_pass": fit_config_pass,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    add_architecture_argument(parser)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest-root",
        type=Path,
        help="default: infer from the supplied configs",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--skip-data-files",
        action="store_true",
        help="verify manifest logic and hashes but skip filesystem existence checks",
    )
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    checks: dict[str, dict[str, Any]] = {}
    config_paths = [
        path if path.is_absolute() else root / path
        for path in (Path(name) for name in args.configs)
    ]
    configured_roots = {str(json.loads(path.read_text())["manifests"]) for path in config_paths}
    resolved_roots = {(Path(name) if Path(name).is_absolute() else root / name).resolve()
                      for name in configured_roots}
    if len(resolved_roots) != 1:
        raise ValueError("configs must use the same frozen manifest directory")
    manifest_root = resolved_roots.pop()
    if args.manifest_root is not None:
        explicit = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
        if explicit.resolve() != manifest_root:
            raise ValueError("explicit manifest directory disagrees with configs")
    configured_sequences = [
        tuple(json.loads(path.read_text()).get("dataset_weights", {})) for path in config_paths
    ]
    datasets = configured_sequences[0] if configured_sequences else ()
    dataset_alignment = (
        bool(datasets)
        and all(value == datasets for value in configured_sequences)
        and not (set(datasets) - set(SUPPORTED_DATASETS))
        and datasets in EXPECTED_WEIGHT_POLICIES
    )
    add_check(
        checks,
        "config_dataset_alignment",
        dataset_alignment,
        {
            "configured_sequences": [list(value) for value in configured_sequences],
            "supported": list(SUPPORTED_DATASETS),
        },
    )
    manifest_summary = audit_manifests(
        manifest_root,
        datasets,
        checks,
        not args.skip_data_files,
    )
    for path in config_paths:
        audit_config(root, path, manifest_root, datasets, checks, architecture=args.architecture)

    model_root = root / "models" / "Wan2.2-Fun-5B-Control"
    asset_expectations = {
        "wan_config": (model_root / "config.json", WAN_CONFIG_SHA256),
        "wan_dit": (
            model_root / "diffusion_pytorch_model.safetensors",
            WAN_DIT_SHA256,
        ),
        "wan_vae": (model_root / "Wan2.2_VAE.pth", WAN_VAE_SHA256),
    }
    asset_hashes: dict[str, str | None] = {}
    for name, (path, expected) in asset_expectations.items():
        digest = sha256(path) if path.is_file() else None
        asset_hashes[name] = digest
        add_check(
            checks,
            name,
            digest == expected,
            {"path": str(path), "sha256": digest, "expected": expected},
        )
    mano_root = root / "assets" / "body_models" / "mano"
    for side in ("LEFT", "RIGHT"):
        path = mano_root / f"MANO_{side}.pkl"
        digest = sha256(path) if path.is_file() else None
        add_check(
            checks,
            f"mano_{side.lower()}",
            digest == MANO_SHA256[side],
            {
                "path": str(path),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": digest,
                "expected": MANO_SHA256[side],
            },
        )
    videox_root = root / "third_party" / "VideoX-Fun"
    videox_tree = source_tree_sha256(videox_root) if videox_root.is_dir() else None
    add_check(
        checks,
        "videox_fun_source_tree",
        videox_tree == VIDEOX_FUN_TREE_SHA256,
        {
            "path": str(videox_root),
            "upstream_commit": VIDEOX_FUN_COMMIT,
            "tree_sha256": videox_tree,
            "expected_tree_sha256": VIDEOX_FUN_TREE_SHA256,
        },
    )

    modules = ["av", "numpy", "projectaria_tools", "smplx", "torch"]
    packages = ["av", "numpy", "projectaria-tools", "smplx", "torch"]
    available = {name: util.find_spec(name) is not None for name in modules}
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    add_check(
        checks,
        "python_dependencies",
        all(available.values()),
        {"modules": available, "versions": versions},
    )

    try:
        import torch

        gpu_detail = {
            "count": torch.cuda.device_count(),
            "names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        }
        gpu_pass = (
            gpu_detail["count"] == 8
            and gpu_detail["bf16_supported"]
            and all("A800" in name for name in gpu_detail["names"])
        )
    except Exception as error:  # pragma: no cover - diagnostic path
        gpu_detail = {"error": f"{type(error).__name__}: {error}"}
        gpu_pass = False
    add_check(checks, "eight_a800_bf16_gpus", gpu_pass, gpu_detail)

    free_bytes = shutil.disk_usage(root).free
    minimum_disk_bytes = 150 * 2**30
    add_check(
        checks,
        "checkpoint_disk_budget",
        free_bytes >= minimum_disk_bytes,
        {"free_bytes": free_bytes, "minimum_bytes": minimum_disk_bytes},
    )
    status = git_value(root, "status", "--porcelain")
    repository = git_value(root, "rev-parse", "--show-toplevel")
    own_repository = bool(repository) and Path(repository).resolve() == root.resolve()
    committed = bool(git_value(root, "rev-parse", "HEAD")) if own_repository else False
    add_check(
        checks,
        "git_clean",
        own_repository and committed and not status if args.require_clean_git else True,
        {"required": args.require_clean_git, "own_repository": own_repository,
         "committed": committed, "status": status.splitlines()},
    )

    ready = all(value["pass"] for value in checks.values())
    output = {
        "format": "handprism-dataset-mixture-readiness-v2-clean",
        "architecture": args.architecture,
        "ready": ready,
        "datasets": list(datasets),
        "wan_revision": WAN_REVISION,
        "checks": checks,
        "manifest_summary": manifest_summary,
    }
    payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        target = args.output if args.output.is_absolute() else root / args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload)
    print(payload, end="")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
