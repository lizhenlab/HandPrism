"""Fail-closed ARCTIC/HOT3D-only policy, checked before opening manifests."""

from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = ("arctic", "hot3d")
SPLITS = ("train", "val", "test")
MANIFEST_NAMES = {f"{name}_{split}.jsonl" for name in SUPPORTED_DATASETS for split in SPLITS}


def require_dataset(name: str) -> str:
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset {name!r}; only ARCTIC/HOT3D are allowed")
    return name


def allowed_path(path: str | Path) -> Path:
    candidate = Path(path)
    if "egodex" in str(candidate).lower():
        raise ValueError("forbidden data path; no data was opened")
    resolved = candidate.resolve()
    if "egodex" in str(resolved).lower():
        raise ValueError("forbidden symlink target; no data was opened")
    return candidate


def manifest_path(path: str | Path) -> Path:
    candidate = allowed_path(path)
    if candidate.name not in MANIFEST_NAMES:
        raise ValueError(f"unsupported manifest filename: {candidate.name}")
    return candidate


def validate_record(record: dict[str, Any], manifest: Path) -> None:
    dataset = require_dataset(str(record.get("dataset")))
    if not manifest.name.startswith(dataset + "_"):
        raise ValueError("manifest filename and record dataset disagree")
    expected_split = manifest.stem.split("_")[-1]
    if record.get("split", expected_split) != expected_split:
        raise ValueError("manifest filename and record split disagree")
    for key, value in record.items():
        if isinstance(value, str) and "egodex" in value.lower():
            raise ValueError(f"forbidden data reference in manifest field {key}")
    for key in ("root", "recording_root"):
        if key in record:
            allowed_path(record[key])
    if "sequence" in record and (
        Path(record["sequence"]).is_absolute() or ".." in Path(record["sequence"]).parts
    ):
        raise ValueError("sequence must stay within its dataset root")
