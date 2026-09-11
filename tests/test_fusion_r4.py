"""Pre-training regressions: fallback gradients, unsafe GT, ROI and frozen val."""
from __future__ import annotations

import copy
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

from handprism.architectures import FUSION, CORE, CHECKPOINT_FORMAT, validate_checkpoint_identity
from handprism.config import LossWeights
from handprism.data.difficulty import (select_validation_windows, stratified_order,
    validation_rank_limit, validate_fusion_index, FUSION_INDEX_VERSION)
from handprism.data.schema import MANIFEST_SCHEMA
from handprism.fusion import FusionConfig, LocalRGB
from handprism.fusion_runtime import (fusion_config_from_json, loss_weights_from_json,
    training_diagnostics, decoder_gradient_norms)
from handprism.losses import HandPrismLoss
from handprism.mano import ToyMano
from handprism.model import HandPrismModel
from handprism.training import prediction_from_output, target_from_batch
from scripts.train import load_config, solver_config_from_json, manifest_report
from test_fusion_r3 import batch_data, tiny

ROOT = Path(__file__).resolve().parents[1]


def setup_model(solver="kfree", seed=0):
    torch.manual_seed(seed)
    config = load_config(ROOT/f"configs/handprism_fusion_{solver}.json")
    options = replace(fusion_config_from_json(config), local_resolution=16, local_chunk_size=3)
    model = HandPrismModel(ToyMano(), tiny(), solver_config_from_json(config), architecture=FUSION,
                           fusion_config=options).train()
    batch, _ = batch_data()
    output = model(torch.randn(1, 2, 4, 4, 24), target_frames=5, solver=solver,
        intrinsics=batch["intrinsics"] if solver == "standard" else None,
        image_size=batch["image_size"], rgb_high=batch["rgb_high"], optimizer_step=0)
    return config, options, model, batch, output


@pytest.mark.parametrize("solver", ["standard", "kfree"])
@pytest.mark.parametrize("seed", [0, 1, 2, 64])
def test_depth_receives_gradient_at_cold_start_even_when_all_hands_fallback(solver, seed):
    config, options, model, batch, output = setup_model(solver, seed)
    target = target_from_batch(batch, 4, 4)
    criterion = HandPrismLoss(loss_weights_from_json(config), options)
    loss = criterion(prediction_from_output(output), target, batch["intrinsics"], batch["image_size"],
                     solver=solver, optimizer_step=0, camera_fit_config=solver_config_from_json(config))
    if solver == "kfree":
        assert output.pnp.used_fallback.all()  # Natural zero-ray initialization, no forced gate.
    loss["total"].backward()
    norm = decoder_gradient_norms(model.decoder)["gradient/camera_head"]
    assert torch.isfinite(norm) and norm > 0
    assert loss["log_depth"] > 0
    diagnostics = training_diagnostics(output, batch)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    if solver == "kfree":
        assert diagnostics["wrist_prior_fraction"] == 1


@pytest.mark.parametrize("fill", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_undefined_unsupervised_annotations_have_finite_forward_and_backward(fill, all_invalid):
    config, options, model, batch, output = setup_model()
    target = target_from_batch(batch, 4, 4)
    # Vary whole-hand and empty-clip masks. Predictions remain finite and live.
    start = 0 if all_invalid else 1
    for name in ("valid_hand", "valid_mano", "valid_joints_3d", "valid_joints_2d", "visibility_valid"):
        value = getattr(target, name).clone()
        value[:, :, start:] = False
        setattr(target, name, value)
    for name in ("global_rotation", "articulation", "betas", "joints_root", "joints_camera",
                 "translation", "joints_2d", "visibility"):
        value = getattr(target, name).float().clone()
        value[:, :, start:] = fill
        setattr(target, name, value)
    target.valid_ray = torch.zeros_like(target.valid_ray)
    target.ray_field = torch.full_like(target.ray_field, fill)
    before = target.global_rotation.clone()
    criterion = HandPrismLoss(loss_weights_from_json(config), options)
    losses = criterion(prediction_from_output(output), target, batch["intrinsics"], batch["image_size"],
                       solver="kfree", optimizer_step=2500, camera_fit_config=solver_config_from_json(config))
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.testing.assert_close(target.global_rotation, before, equal_nan=True)  # no input mutation
    if all_invalid:
        for key in ("rotation_matrix", "rotation_geodesic", "log_depth", "wrist_prior", "ray",
                    "joints_root", "joints_camera", "velocity_error", "reliability"):
            assert losses[key] == 0, key


@pytest.mark.parametrize("name", ["global_rotation", "articulation", "joints_root", "joints_camera",
                                  "joints_2d", "ray_field", "betas", "existence", "visibility"])
def test_valid_nonfinite_annotations_are_rejected(name):
    config, options, _, batch, output = setup_model()
    target = target_from_batch(batch, 4, 4)
    value = getattr(target, name).float().clone()
    value.reshape(-1)[0] = float("nan")
    setattr(target, name, value)
    with pytest.raises(ValueError, match="non-finite valid ground truth"):
        HandPrismLoss(loss_weights_from_json(config), options)(prediction_from_output(output), target,
            batch["intrinsics"], batch["image_size"])


def test_masks_cannot_hide_nonfinite_model_predictions():
    config, options, _, batch, output = setup_model()
    prediction = prediction_from_output(output)
    prediction.log_depth = torch.full_like(prediction.log_depth, float("nan"))
    target = target_from_batch(batch, 4, 4)
    target.valid_hand = torch.zeros_like(target.valid_hand)
    with pytest.raises(RuntimeError, match="non-finite model prediction"):
        HandPrismLoss(loss_weights_from_json(config), options)(prediction, target,
            batch["intrinsics"], batch["image_size"])


def test_invalid_gt_does_not_pollute_validation_or_test_metrics():
    from handprism.training import scoring_batch
    from handprism.evaluator import EvaluationAccumulator, score_batch, validate_metric_result
    _, _, model, batch, output = setup_model()
    batch = copy.deepcopy(batch)
    for name in ("valid_hand", "valid_mano", "valid_joints_3d", "valid_joints_2d", "visibility_valid"):
        batch[name] = batch[name].clone()
        batch[name][:, :, 1] = False
    for name in ("global_rotation", "articulation", "betas", "joints_root", "joints_camera",
                 "translation", "joints_2d", "visibility"):
        batch[name] = batch[name].float().clone()
        batch[name][:, :, 1] = float("nan")
    cleaned = scoring_batch(batch, 4, 4)
    assert torch.isnan(batch["global_rotation"][:, :, 1]).all()
    with torch.no_grad():
        canonical, vertices = model.mano(cleaned["global_rotation"], cleaned["articulation"], cleaned["betas"])
        accumulator = EvaluationAccumulator()
        score_batch(accumulator, output, cleaned, vertices, canonical[0, 0], solver="kfree",
                    gt_mano_translation=cleaned["translation"]-model.mano.root_offset(cleaned["betas"]))
    validate_metric_result(accumulator.finalize())


def test_log_depth_uses_positive_wrist_z_not_range_or_invalid_joints():
    config, options, _, batch, output = setup_model()
    weights = LossWeights(**{field.name: .5 if field.name == "log_depth" else 0 for field in fields(LossWeights)})
    prediction = prediction_from_output(output)
    target = target_from_batch(batch, 4, 4)
    target.joints_camera = target.joints_camera.clone()
    target.joints_camera[..., 0, 0] = 1000  # Must not change a Z-depth target.
    target.joints_camera[:, 0, :, 0, 2] = -.2
    prediction.log_depth = torch.full_like(prediction.log_depth, .7).log().detach().requires_grad_()
    losses = HandPrismLoss(weights, options)(prediction, target, batch["intrinsics"], batch["image_size"])
    assert losses["log_depth"] == 0
    losses["total"].backward()
    assert torch.equal(prediction.log_depth.grad, torch.zeros_like(prediction.log_depth))


def test_depth_loss_is_mandatory_only_with_wrist_prior_and_core_is_frozen():
    config = load_config(ROOT/"configs/handprism_fusion_standard.json")
    config["loss_weights"]["log_depth"] = 0
    with pytest.raises(ValueError, match="independent log_depth"):
        loss_weights_from_json(config)
    core = load_config(ROOT/"configs/handprism_core_standard.json", architecture=CORE)
    assert loss_weights_from_json(core).log_depth == 0
    core["loss_weights"] = {"log_depth": .5}
    with pytest.raises(ValueError, match="Core loss weights are frozen"):
        loss_weights_from_json(core)


def test_roi_curriculum_selects_whole_boxes_with_clip_coherent_choices_and_jitter():
    torch.manual_seed(6)
    local = LocalRGB(8, FusionConfig(local_rgb=True, local_resolution=16)).train()
    anchors = torch.full((12, 5, 2, 21, 2), .8)
    teacher = torch.full_like(anchors, .1)
    hand = torch.zeros(12, 5, 2, 8)
    rgb = torch.zeros(12, 3, 5, 32, 32)
    valid = torch.ones(teacher.shape[:-1], dtype=torch.bool)
    _, good, bounds = local(rgb, anchors, hand, teacher, valid, .5, torch.full((12, 2), 32.))
    centers = (bounds[..., :2] + bounds[..., 2:]) / 2
    chosen = centers[..., 0] < .2
    assert chosen.any() and (~chosen).any()
    assert good.all()
    torch.testing.assert_close(centers, centers[:, :1].expand_as(centers))
    assert ((centers[..., 0] < .2) | (centers[..., 0] > .7)).all()  # Never midpoint/background.
    rng = torch.get_rng_state()
    _, _, disabled = local(rgb, anchors, hand, teacher, valid, 0, torch.full((12, 2), 32.))
    assert torch.equal(torch.get_rng_state(), rng)
    _, _, predicted = local(rgb, anchors, hand, image_size=torch.full((12, 2), 32.))
    assert torch.equal(disabled, predicted)
    anchors.fill_(float("nan"))
    _, good, _ = local(rgb, anchors, hand, teacher, valid, 1, torch.full((12, 2), 32.))
    assert good.all()  # Valid teacher does not inherit invalid predicted ROI flag.


@pytest.mark.parametrize("world", [1, 2, 3, 8, 32])
def test_fast_validation_is_the_same_global_prefix_at_any_world_size(world):
    global_order = list(range(317))
    selected = []
    for rank in range(world):
        limit = validation_rank_limit(16, rank, world)
        selected.extend(global_order[rank::world][:limit])
        assert validation_rank_limit(0, rank, world) is None
    assert sorted(selected) == list(range(16))


def test_temporal_and_stratified_validation_is_deterministic_and_nonoverlapping():
    candidates = [dict(start_frame=i*81, frames=81, recording_id="val", split_group="p1",
                       validation_strata=["fast"] if i == 11 else []) for i in range(100)]
    chosen = select_validation_windows(candidates)
    assert chosen == select_validation_windows(list(reversed(candidates)))
    assert len(chosen) == 12 and chosen[0]["start_frame"] == 0 and chosen[-1]["start_frame"] == 99*81
    assert any(row["validation_strata"] == ["fast"] for row in chosen)
    assert select_validation_windows(candidates[:2]) == candidates[:2]
    with pytest.raises(ValueError, match="overlap"):
        select_validation_windows(candidates[:2]+candidates[:1])


def test_old_fusion_identity_and_per_rank_validation_budget_are_rejected():
    config = load_config(ROOT/"configs/handprism_fusion_standard.json")
    assert config["implementation_id"] == "handprism-fusion-r4"
    checkpoint = dict(format=CHECKPOINT_FORMAT, architecture=FUSION,
                      implementation_id="handprism-fusion-r3", config=config)
    with pytest.raises(ValueError, match="implementation mismatch"):
        validate_checkpoint_identity(checkpoint, config, architecture=FUSION)
    from scripts.train import validate_config
    config["validation_batches_per_dataset"] = 8
    with pytest.raises(ValueError, match="global validation_clips"):
        validate_config(config)


def test_fast_stratification_rotates_recordings_before_reusing_them():
    records = [dict(recording_id=str(r), split_group="participant:val", validation_strata=[],
                    start_frame=w*81) for r in range(20) for w in range(12)]
    order = stratified_order(records)
    assert len({records[i]["recording_id"] for i in order[:16]}) == 16
    assert sorted(order) == list(range(len(records)))


def write_source_index(path):
    from scripts.build_manifests import HOT3D_REQUIRED_MASKS
    path.mkdir()
    report = dict(version=MANIFEST_SCHEMA, selected_datasets=["arctic", "hot3d"],
                  frames_per_window=81, datasets={})
    for name in report["selected_datasets"]:
        report["datasets"][name] = {}
        for split in ("train", "val", "test"):
            count = 2 if split == "val" else 1
            rows = []
            for i in range(count):
                record = dict(dataset=name, split=split, recording_id=f"{split}-{i}",
                    split_group=f"participant:{split}", frames=81, num_frames=81*24,
                    capabilities=["mano"], image_offset=0, sequence=f"{split}/{i}",
                    root=str(path.parent/"arctic"), recording_root=str(path.parent/f"{split}-{i}"),
                    usable_start=0, usable_stop=81*24, required_masks=list(HOT3D_REQUIRED_MASKS),
                    mask_stats={"certified": 1})
                if split == "train":
                    record.update(start_min=0, start_max=162)
                else:
                    record["start_frame"] = 81
                rows.append(record)
            payload = "".join(json.dumps(row)+"\n" for row in rows).encode()
            (path/f"{name}_{split}.jsonl").write_bytes(payload)
            report["datasets"][name][split] = dict(rows=count, recordings=count, split_groups=1,
                sha256=hashlib.sha256(payload).hexdigest())
    (path/"split_report.json").write_text(json.dumps(report))


def test_index_builder_publishes_new_multival_only_and_keeps_test_bytes_and_groups(tmp_path, monkeypatch):
    from scripts import prepare_fusion_index as builder
    source, output = tmp_path/"source", tmp_path/"fusion"
    write_source_index(source)
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    decoded = []

    def geometry(record, start, mano):
        assert record["split"] in ("train", "val")  # Fails if any test geometry is touched.
        decoded.append((record["dataset"], record["split"], record["recording_id"], start))
        return start

    monkeypatch.setattr(builder, "geometry_sample", geometry)
    monkeypatch.setattr(builder, "window_strata", lambda start: ["edge"] if start % 162 else ["fast"])
    monkeypatch.setattr(builder, "hot3d_valid_ranges", lambda root: (81*24, [(0, 81*24)], {"certified": 1}))
    monkeypatch.setattr(sys, "argv", ["index", "--source", str(source), "--output", str(output),
                                     "--mano-model", str(tmp_path/"mano")])
    builder.main()
    assert original == {p.name: p.read_bytes() for p in source.iterdir()}
    audit = manifest_report(output, ("arctic", "hot3d"))
    report = audit["split_report"]
    assert report["fusion_index"]["version"] == FUSION_INDEX_VERSION
    records = {}
    for name in ("arctic", "hot3d"):
        assert (output/f"{name}_test.jsonl").read_bytes() == original[f"{name}_test.jsonl"]
        assert audit["manifests"][f"{name}_val.jsonl"]["rows"] == 24
        assert audit["manifests"][f"{name}_val.jsonl"]["recordings"] == 2
        records[name] = {split: [json.loads(line) for line in (output/f"{name}_{split}.jsonl").read_text().splitlines()]
                         for split in ("train", "val")}
    validate_fusion_index(report, records, 16)
    with pytest.raises(ValueError, match="more windows"):
        validate_fusion_index(report, records, 24)
    broken = copy.deepcopy(records)
    broken["arctic"]["val"][1]["start_frame"] = broken["arctic"]["val"][0]["start_frame"]
    with pytest.raises(ValueError, match="overlapping"):
        validate_fusion_index(report, broken)
    broken = copy.deepcopy(records)
    broken["hot3d"]["val"][0]["start_frame"] = 100000
    with pytest.raises(ValueError, match="certified ranges"):
        validate_fusion_index(report, broken)


def test_hot3d_val_bounds_are_recovered_from_masks_and_never_guessed(monkeypatch):
    from scripts import prepare_fusion_index as builder
    record = dict(dataset="hot3d", split="val", start_frame=10, num_frames=500,
                  recording_root="data/hot3d/val-record", required_masks=list(builder.HOT3D_REQUIRED_MASKS),
                  mask_stats={"certified": 1})
    monkeypatch.setattr(builder, "hot3d_valid_ranges", lambda root: (500, [(10, 100), (200, 290)], {"certified": 1}))
    assert builder.validation_ranges(record) == [[10, 100], [200, 290]]
    record["start_frame"] = 110  # Source start in a masked-out gap must stop construction.
    with pytest.raises(ValueError, match="outside certified"):
        builder.validation_ranges(record)
    record["split"] = "test"
    with pytest.raises(ValueError, match="only accepts val"):
        builder.validation_ranges(record)
