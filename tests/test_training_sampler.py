from __future__ import annotations

import json

from scripts.train import (
    DeterministicDrawBatchSampler,
    consumed_dataset_batches,
    dataset_for_step,
    load_config,
)
import pytest
from handprism.architectures import FUSION, architecture_spec


def test_draw_sampler_preserves_requested_per_gpu_batch_and_rank_disjointness() -> None:
    rank_zero = DeterministicDrawBatchSampler(101, 3, 0, 2, 17, "arctic")
    rank_one = DeterministicDrawBatchSampler(101, 3, 1, 2, 17, "arctic")
    batch_zero = next(iter(rank_zero))
    batch_one = next(iter(rank_one))
    assert len(batch_zero) == 3
    assert len(batch_one) == 3
    assert {index for index, _ in batch_zero}.isdisjoint({index for index, _ in batch_one})


def test_draw_sampler_resume_reconstructs_exact_batch_stream() -> None:
    original = DeterministicDrawBatchSampler(17, 2, 1, 3, 91, "hot3d")
    iterator = iter(original)
    batches = [next(iterator) for _ in range(6)]
    resumed = DeterministicDrawBatchSampler(17, 2, 1, 3, 91, "hot3d")
    resumed.set_start_batch(4)
    assert next(iter(resumed)) == batches[4]


def test_consumed_dataset_batches_matches_microstep_count() -> None:
    config = {
        "seed": 5,
        "gradient_accumulation": 8,
        "dataset_weights": {"arctic": 0.4375, "hot3d": 0.5625},
    }
    counts = consumed_dataset_batches(config, completed_steps=13)
    assert sum(counts.values()) == 13 * 8
    for step in range(13):
        selected = dataset_for_step(config, step)
        assert counts[selected] >= 8


def test_load_config_accepts_arctic_hot3d_only(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "seed": 1,
                "solver": "standard",
                "steps": 20_000,
                "batch_size_per_gpu": 1,
                "gradient_accumulation": 8,
                "dataset_weights": {"arctic": 0.4375, "hot3d": 0.5625},
                "dataset_roots": {"arctic": "/arctic", "hot3d": "/hot3d"},
                "manifests": "data/manifests/two_dataset_v2_clean",
                "architecture": FUSION,
                "implementation_id": architecture_spec(FUSION).implementation_id,
                "architecture_contract": architecture_spec(FUSION).contract,
                "geometry_dtype": "float32",
                "loss_reduction": "per_clip",
                "decoder": {"anchor_offset_cells": 0.0},
                "fusion": {},
            }
        )
    )
    assert tuple(load_config(path)["dataset_weights"]) == ("arctic", "hot3d")


@pytest.mark.parametrize("weight", [0.0, -0.1, float("inf"), float("nan")])
def test_load_config_rejects_invalid_dataset_weight(tmp_path, weight) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "seed": 1,
                "solver": "standard",
                "steps": 20_000,
                "batch_size_per_gpu": 1,
                "gradient_accumulation": 8,
                "dataset_weights": {"arctic": weight, "hot3d": 1.0 - weight},
            }
        )
    )
    with pytest.raises(ValueError, match="finite and positive"):
        load_config(path)
