from __future__ import annotations

from pathlib import Path

from handprism.data.dataset import HandPrismWindowDataset
from scripts.build_manifests import (
    FRAMES,
    HOT3D_REQUIRED_MASKS,
    fixed_start,
    fixed_start_in_ranges,
    hot3d_valid_ranges,
    validation_groups,
)
import pytest


def _write_mask(path: Path, values: list[bool]) -> None:
    rows = ["timestamp[ns],stream_id,mask"]
    rows.extend(
        f"{index + 1},214-1,{'True' if value else 'False'}" for index, value in enumerate(values)
    )
    path.write_text("\n".join(rows) + "\n")


def test_fixed_starts_honor_nonzero_valid_bounds() -> None:
    assert 10 <= fixed_start("sequence", 10, 37) <= 37
    selected = fixed_start_in_ranges("recording", ((20, 20 + FRAMES), (200, 300)))
    assert selected == 20 or 200 <= selected <= 300 - FRAMES
    assert len(validation_groups((f"session-{i}" for i in range(40)), "test")) == 2


def test_dataset_draw_never_crosses_hot3d_invalid_gap() -> None:
    dataset = HandPrismWindowDataset.__new__(HandPrismWindowDataset)
    dataset.training = True
    dataset.hard_window_fraction = 0.
    dataset.frames = FRAMES
    record = {"valid_ranges": [[0, 100], [150, 250]]}
    starts = {dataset._start(record, draw_seed) for draw_seed in range(40)}
    assert starts
    assert all(0 <= start <= 19 or 150 <= start <= 169 for start in starts)


def test_hot3d_false_mask_splits_contiguous_eligible_ranges(tmp_path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    values = [True] * 200
    for name in HOT3D_REQUIRED_MASKS:
        current = values.copy()
        if name == "mask_qa_pass":
            current[100] = False
        _write_mask(masks / f"{name}.csv", current)
    frame_count, ranges, stats = hot3d_valid_ranges(tmp_path)
    assert frame_count == 200
    assert ranges == [(0, 100), (101, 200)]
    assert stats["all_required_masks_true"] == 199
