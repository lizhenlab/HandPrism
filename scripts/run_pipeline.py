#!/usr/bin/env python3
"""Run HandPrism readiness, training and evaluation for both camera modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
from typing import Any
from dreamhand.completion import validated_metrics
from dreamhand.data.dataset import read_jsonl
from dreamhand.data.policy import SUPPORTED_DATASETS, allowed_path
from dreamhand.paths import v3_run_path
from dreamhand.architectures import add_architecture_argument, architecture_spec
from scripts.train import load_config


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class Supervisor:
    def __init__(self, root: Path, control_dir: Path, world_size: int, architecture: str) -> None:
        architecture_spec(architecture)
        self.architecture = architecture
        self.root = root
        self.control_dir = control_dir
        self.world_size = world_size
        existing = self.control_dir / "state.json"
        if existing.exists() and json.loads(existing.read_text()).get("architecture") != architecture:
            raise ValueError("supervisor directory belongs to a different or unnamed architecture")
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.events = control_dir / "supervisor.jsonl"
        self.state = control_dir / "state.json"

    def event(self, event_type: str, **values: Any) -> None:
        row = {"type": event_type, "time_unix": time.time(), "architecture": self.architecture, **values}
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    def set_state(self, status: str, **values: Any) -> None:
        atomic_json(
            self.state,
            {"status": status, "time_unix": time.time(), "architecture": self.architecture, **values},
        )

    def run(self, phase: str, command: list[str]) -> None:
        log_path = self.control_dir / f"{phase}.log"
        self.set_state("running", phase=phase, command=command, log=str(log_path))
        self.event("phase_start", phase=phase, command=command, log=str(log_path))
        started = time.monotonic()
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=self.root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        seconds = time.monotonic() - started
        self.event(
            "phase_end",
            phase=phase,
            returncode=result.returncode,
            seconds=seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(f"phase {phase} exited with status {result.returncode}")

    def distributed(self, script: str, *arguments: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={self.world_size}",
            script,
            "--architecture",
            self.architecture,
            *arguments,
        ]


def checkpoint_step(run_dir: Path) -> tuple[int, Path] | None:
    latest = run_dir / "checkpoints" / "latest.json"
    if not latest.is_file():
        return None
    value = json.loads(latest.read_text())
    path = latest.parent / value["path"]
    if not path.is_file():
        raise FileNotFoundError(f"latest checkpoint pointer is broken: {path}")
    return int(value["step"]), path


def resource_preflight(root: Path, experiments: tuple) -> dict[str, float]:
    """Never overlap a new job with occupied GPUs or knowingly exhaust disk."""
    gib = 1024**3
    checkpoint_budget = 0.0
    for _, _, run_dir in experiments:
        latest = checkpoint_step(run_dir)
        step, size = (latest[0], latest[1].stat().st_size) if latest else (0, 1.6 * gib)
        checkpoint_budget += max(0, 40 - step // 500) * size
    free = shutil.disk_usage(root).free
    # Keep 80 GiB beyond future checkpoints, so a run that just fits does not
    # predictably enter the monitor's emergency zone before final evaluation.
    required = checkpoint_budget + 80 * gib
    if free < required:
        raise RuntimeError(
            f"insufficient disk for v3: {free / gib:.1f} GiB free; {required / gib:.1f} GiB required; no files deleted"
        )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if gpu.returncode or gpu.stdout.strip():
        raise RuntimeError(
            "GPU resource preflight failed or compute jobs already running; no job stopped"
        )
    return {"free_gib": free / gib, "required_gib": required / gib}


def evaluation_directory(
    run_dir: Path,
    step: int,
    *,
    architecture: str,
    solver: str | None = None,
    checkpoint: Path | None = None,
    expected_counts: dict[str, int] | None = None,
) -> Path:
    base = run_dir / f"evaluation_step_{step:06d}"
    if not base.exists():
        return base
    if solver is not None and checkpoint is not None and expected_counts is not None:
        for directory in [base, *sorted(run_dir.glob(f"{base.name}_attempt_*"))]:
            try:
                validated_metrics(
                    directory,
                    architecture=architecture,
                    solver=solver,
                    step=step,
                    checkpoint=checkpoint,
                    expected_counts=expected_counts,
                )
                return directory
            except (OSError, ValueError, TypeError, KeyError):
                pass  # Preserve failed/incomplete attempts; never overwrite them.
    index = 2
    while (candidate := run_dir / f"evaluation_step_{step:06d}_attempt_{index}").exists():
        index += 1
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    add_architecture_argument(parser)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument(
        "--control-dir",
        type=Path,
    )
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/manifests/two_dataset_v2_clean")
    )
    parser.add_argument("--standard-config", type=Path)
    parser.add_argument(
        "--kfree-config",
        type=Path,
    )
    parser.add_argument(
        "--standard-run-dir",
        type=Path,
    )
    parser.add_argument(
        "--kfree-run-dir",
        type=Path,
    )
    args = parser.parse_args()
    root = args.root.resolve() if args.root is not None else Path(__file__).resolve().parents[1]
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("supervisor cannot write into another workspace")
    stem = args.architecture.replace("-", "_")
    control_dir = v3_run_path(root, args.control_dir or Path(f"runs/{stem}_full"))
    standard_run_dir = v3_run_path(root, args.standard_run_dir or Path(f"runs/{stem}_standard"))
    kfree_run_dir = v3_run_path(root, args.kfree_run_dir or Path(f"runs/{stem}_kfree"))
    args.standard_config = args.standard_config or Path(f"configs/{stem}_standard.json")
    args.kfree_config = args.kfree_config or Path(f"configs/{stem}_kfree.json")

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    experiments = (
        (
            "standard",
            resolve(args.standard_config),
            standard_run_dir,
        ),
        (
            "kfree",
            resolve(args.kfree_config),
            kfree_run_dir,
        ),
    )
    for solver, path, _ in experiments:
        checked = load_config(path, architecture=args.architecture)
        if checked["solver"] != solver:
            raise ValueError("supervisor phase/config solver mismatch")
    supervisor = Supervisor(root, control_dir, args.world_size, args.architecture)
    try:
        for _, _, directory in experiments:
            allowed_path(directory)
        supervisor.run(
            "readiness",
            [
                sys.executable,
                "scripts/check_readiness.py",
                "--architecture",
                args.architecture,
                "--root",
                str(root),
                "--require-clean-git",
                "--manifest-root",
                str(resolve(args.manifest_root)),
                "--configs",
                str(resolve(args.standard_config)),
                str(resolve(args.kfree_config)),
                "--output",
                str(control_dir / "readiness.json"),
            ],
        )
        expected_counts = {
            dataset: len(read_jsonl(resolve(args.manifest_root) / f"{dataset}_test.jsonl"))
            for dataset in SUPPORTED_DATASETS
        }
        if expected_counts != {"arctic": 291, "hot3d": 437}:
            raise RuntimeError("full test requires the frozen 291/437 segment split")
        results: dict[str, str] = {}
        for solver, config, run_dir in experiments:
            latest = checkpoint_step(run_dir)
            if latest is None or latest[0] < 20_000:
                supervisor.event("resource_preflight", **resource_preflight(root, experiments))
                supervisor.run(
                    f"train_{solver}",
                    supervisor.distributed(
                        "scripts/train.py",
                        "--config",
                        str(config),
                        "--run-dir",
                        str(run_dir),
                        "--resume",
                        "auto",
                    ),
                )
                latest = checkpoint_step(run_dir)
            if latest is None or latest[0] != 20_000:
                raise RuntimeError(f"{solver} did not produce the step-20000 checkpoint")
            step, checkpoint = latest
            evaluation = evaluation_directory(
                run_dir, step, architecture=args.architecture,
                solver=solver, checkpoint=checkpoint, expected_counts=expected_counts
            )
            if (evaluation / "metrics.json").is_file():
                supervisor.event(
                    "phase_skip",
                    phase=f"evaluate_{solver}",
                    reason="metrics, full test counts and checkpoint hash validated",
                    output=str(evaluation),
                )
            else:
                supervisor.event("resource_preflight", **resource_preflight(root, experiments))
                supervisor.run(
                    f"evaluate_{solver}",
                    supervisor.distributed(
                        "scripts/evaluate.py",
                        "--config",
                        str(config),
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(evaluation),
                    ),
                )
            validated_metrics(
                evaluation,
                architecture=args.architecture,
                solver=solver,
                step=step,
                checkpoint=checkpoint,
                expected_counts=expected_counts,
            )
            results[solver] = str(evaluation / "metrics.json")
        supervisor.set_state("complete", phase="all", results=results)
        supervisor.event("pipeline_complete", results=results)
    except BaseException as error:
        supervisor.set_state("failed", error=f"{type(error).__name__}: {error}")
        supervisor.event("pipeline_failed", error=f"{type(error).__name__}: {error}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
