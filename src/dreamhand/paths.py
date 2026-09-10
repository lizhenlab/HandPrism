"""Keep new experiment outputs isolated from preserved model runs."""

from pathlib import Path

from .data.policy import allowed_path


def v3_run_path(root: Path, path: Path) -> Path:
    root = root.resolve()
    candidate = allowed_path(path if path.is_absolute() else root / path).resolve()
    runs = root / "runs"
    if not candidate.is_relative_to(runs) or candidate == runs:
        raise ValueError("output must stay inside this workspace's runs/ directory")
    if any(
        "v2_clean" in part or part.startswith("three_dataset")
        for part in candidate.relative_to(runs).parts
    ):
        raise ValueError("preserved and historical run directories are protected")
    return candidate
