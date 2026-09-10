"""Manifest-backed ARCTIC/HOT3D loading and homogeneous collation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .arctic import load_arctic_window
from .contract import DreamHandSample
from .hot3d import load_hot3d_window
from .policy import manifest_path, require_dataset, validate_record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = manifest_path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            validate_record(value, path)
            records.append(value)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


class DreamHandWindowDataset(Dataset[DreamHandSample]):
    def __init__(
        self,
        manifest: str | Path,
        *,
        mano_model_path: str | Path,
        training: bool,
        frames: int = 81,
    ) -> None:
        self.manifest = Path(manifest)
        self.records = read_jsonl(self.manifest)
        self.mano_model_path = Path(mano_model_path)
        self.training = training
        self.frames = frames
        datasets = {str(record.get("dataset")) for record in self.records}
        if len(datasets) != 1:
            raise ValueError(f"one manifest must contain one dataset, got {datasets}")
        self.dataset_name = datasets.pop()

    def __len__(self) -> int:
        return len(self.records)

    def _start(self, record: dict[str, Any], draw_seed: int | None = None) -> int:
        if not self.training:
            return int(record["start_frame"])
        if "valid_ranges" in record:
            ranges = [(int(start), int(stop)) for start, stop in record["valid_ranges"]]
            counts = [stop - start - self.frames + 1 for start, stop in ranges]
            total = sum(counts)
            if total <= 0 or any(count <= 0 for count in counts):
                raise ValueError("valid_ranges contain no complete training window")
            selected = (
                draw_seed % total if draw_seed is not None else int(torch.randint(total, ()).item())
            )
            for (start, _), count in zip(ranges, counts):
                if selected < count:
                    return start + selected
                selected -= count
            raise AssertionError("unreachable valid-range selection")
        lower = int(record.get("start_min", record.get("start_frame", 0)))
        upper = int(
            record.get(
                "start_max",
                int(record["num_frames"]) - self.frames,
            )
        )
        if upper < lower:
            raise ValueError(f"invalid training window range [{lower},{upper}]")
        if upper == lower:
            return lower
        if draw_seed is not None:
            return lower + draw_seed % (upper - lower + 1)
        return int(torch.randint(lower, upper + 1, ()).item())

    def __getitem__(self, index: int | tuple[int, int]) -> DreamHandSample:
        draw_seed: int | None = None
        if isinstance(index, tuple):
            index, draw_seed = index
        record = self.records[index]
        start = self._start(record, draw_seed)
        dataset = record["dataset"]
        if dataset == "arctic":
            return load_arctic_window(
                record["root"],
                record["sequence"],
                start,
                self.frames,
                mano_model_path=self.mano_model_path,
                image_offset=int(record["image_offset"]),
            )
        if dataset == "hot3d":
            return load_hot3d_window(
                record["recording_root"],
                start,
                self.frames,
                mano_model_path=self.mano_model_path,
                required_masks=tuple(record["required_masks"]),
            )
        raise ValueError(f"unsupported dataset {dataset!r}")


def collate_samples(samples: list[DreamHandSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    names = {sample.dataset for sample in samples}
    for name in names:
        require_dataset(name)
    if len(names) != 1:
        raise ValueError("batches must be homogeneous by dataset")
    tensor_fields = (
        "video",
        "intrinsics",
        "image_size",
        "global_rotation",
        "articulation",
        "betas",
        "translation",
        "joints_root",
        "joints_camera",
        "joints_2d",
        "existence",
        "visibility",
        "valid_hand",
        "valid_mano",
        "valid_joints_3d",
        "valid_joints_2d",
        "valid_ray",
    )
    result: dict[str, Any] = {
        field: torch.stack([getattr(sample, field) for sample in samples])
        for field in tensor_fields
    }
    result["distortion"] = torch.stack(
        [
            sample.distortion if sample.distortion is not None else torch.zeros(8)
            for sample in samples
        ]
    )
    camera_models = {sample.camera_model for sample in samples}
    if len(camera_models) != 1:
        raise ValueError("batches must be homogeneous by camera model")
    result["camera_model"] = camera_models.pop()
    for field in ("camera_parameters", "source_image_size", "gt_ray_field"):
        values = [getattr(sample, field) for sample in samples]
        if all(value is None for value in values):
            result[field] = None
        elif any(value is None for value in values):
            raise ValueError(f"optional camera field {field} must be present for the whole batch")
        else:
            result[field] = torch.stack(values)  # type: ignore[arg-type]
    result["dataset"] = samples[0].dataset
    result["recording_id"] = [sample.recording_id for sample in samples]
    result["frame_indices"] = torch.stack([sample.frame_indices for sample in samples])
    return result
