"""Strict completion evidence for full two-dataset evaluations."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .architectures import architecture_spec


def require_finite_json(value: Any, name: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite JSON value at {name}")
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{name}[{index}]")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_metrics(
    directory: Path, *, architecture: str, solver: str, step: int, checkpoint: Path,
    expected_counts: dict[str, int]
) -> dict[str, Any]:
    report = json.loads((directory / "metrics.json").read_text())
    require_finite_json(report)
    spec = architecture_spec(architecture)
    if (not isinstance(report, dict) or report.get("architecture") != architecture
            or report.get("implementation_id") != spec.implementation_id):
        raise ValueError("evaluation implementation contract mismatch")
    if report.get("solver") != solver or report.get("checkpoint_step") != step:
        raise ValueError("evaluation solver/step mismatch")
    if set(expected_counts) != {"arctic", "hot3d"} or set(report.get("datasets", {})) != set(
        expected_counts
    ):
        raise ValueError("evaluation must contain exactly ARCTIC/HOT3D")
    if not report.get("full_test", False):
        raise ValueError("a diagnostic/subset evaluation cannot complete the full run")
    for dataset, count in expected_counts.items():
        if report["datasets"][dataset].get("segments") != count:
            raise ValueError(f"incomplete {dataset} test segments")
    if report.get("overall", {}).get("segments") != sum(expected_counts.values()):
        raise ValueError("overall test segment count mismatch")
    if not checkpoint.is_file() or report.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("evaluation checkpoint hash mismatch")
    return report
