"""Fusion refinements: real autograd, coordinate/clock contracts, no datasets."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from handprism.architectures import CORE, FUSION, validate_checkpoint_identity, CHECKPOINT_FORMAT
from handprism.camera import project_pinhole
from handprism.config import DecoderConfig, SolverConfig, LossWeights
from handprism.data.augmentation import AugmentationConfig, augment_sample
from handprism.data.difficulty import stratified_order
from handprism.data.dataset import HandPrismWindowDataset
from handprism.decoder import HandPrismDecoder
from handprism.evaluator import EvaluationAccumulator, score_batch, validate_metric_result
from handprism.fusion import FusionConfig, LocalRGB, FusionRefinement
from handprism.fusion_runtime import (fusion_config_from_json, loss_weights_from_json,
    fusion_forward_options, validation_selection_score)
from handprism.losses import HandPrismLoss
from handprism.mano import ToyMano
from handprism.model import HandPrismModel
from handprism.motion import motion_errors
from handprism.ray import mixed_pnp, bearings_from_intrinsics
from handprism.training import prediction_from_output, target_from_batch
from scripts.train import load_config, solver_config_from_json
from scripts.infer import load_clip

ROOT = Path(__file__).resolve().parents[1]


def tiny():
    return DecoderConfig(feature_dim=24, hidden_dim=16, layers=2, heads=4, ffn_dim=32,
                         anchor_offset_cells=1)


def batch_data(frames=5):
    identity = torch.eye(3).expand(1, frames, 2, 3, 3).clone()
    pose = torch.eye(3).expand(1, frames, 2, 15, 3, 3).clone()
    betas = torch.zeros(1, frames, 2, 10)
    joints, vertices = ToyMano()(identity, pose, betas)
    wrist = torch.tensor([.05, -.03, .7]).expand(1, frames, 2, 3).clone()
    camera = joints + wrist[..., None, :]
    k = torch.tensor([[[100., 0, 64], [0, 100., 64], [0, 0, 1]]])
    size = torch.tensor([[128., 128.]])
    uv = project_pinhole(camera, k, size)
    hand = torch.ones(1, frames, 2, dtype=torch.bool)
    valid = hand[..., None].expand(-1, -1, -1, 21)
    batch = dict(dataset="arctic", intrinsics=k, image_size=size, distortion=torch.zeros(1, 8),
        camera_model="pinhole", camera_parameters=None, source_image_size=None, gt_ray_field=None,
        global_rotation=identity, articulation=pose, betas=betas, joints_root=joints, joints_camera=camera,
        translation=wrist, joints_2d=uv, existence=hand, visibility=hand, valid_hand=hand,
        valid_mano=hand, valid_joints_3d=valid, valid_joints_2d=valid, valid_ray=torch.ones(1, dtype=torch.bool),
        timestamps=torch.arange(frames, dtype=torch.float64)[None]/30,
        in_frame=valid, visibility_valid=hand, synthetic_occluded=torch.zeros_like(valid),
        rgb_high=torch.randint(256, (1, 3, frames, 192, 192), dtype=torch.uint8))
    return batch, vertices


@pytest.mark.parametrize("solver", ["standard", "kfree"])
def test_all_modules_forward_backward_and_loss_contract(solver):
    torch.manual_seed(64)
    config = load_config(ROOT/f"configs/handprism_fusion_{solver}.json")
    opts = replace(fusion_config_from_json(config), local_resolution=16, local_chunk_size=3)
    model = HandPrismModel(ToyMano(), tiny(), solver_config_from_json(config),
                           architecture=FUSION, fusion_config=opts).train()
    batch, _ = batch_data()
    criterion = HandPrismLoss(loss_weights_from_json(config), opts)
    features = torch.randn(1, 2, 4, 4, 24, requires_grad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Zero-init residual heads are identity at initialization; a second update
    # is necessary to test gradients reaching their internal encoders.
    for _ in range(2):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(features, target_frames=5, solver=solver,
                intrinsics=batch["intrinsics"] if solver=="standard" else None,
                image_size=batch["image_size"], rgb_high=batch["rgb_high"], optimizer_step=2500)
            losses = criterion(prediction_from_output(output), target_from_batch(batch, 4, 4),
                batch["intrinsics"], batch["image_size"], solver=solver, optimizer_step=2500,
                camera_fit_config=solver_config_from_json(config))
        assert all(torch.isfinite(v) for v in losses.values())
        assert losses["acceleration"] >= 0 and criterion.weights.acceleration == 0
        losses["total"].backward()
        assert torch.isfinite(features.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if _ == 1:
            for name in ("final_spatial.q.weight", "refinement.joint_pose.2.weight",
                         "refinement.wrist.2.weight", "refinement.quality.weight", "refinement.local.encoder.0.weight"):
                gradient = dict(model.decoder.named_parameters())[name].grad
                assert gradient is not None and gradient.abs().sum() > 0, name
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert output.decoder.anchors_2d.dtype == torch.float32
    assert output.decoder.wrist_prior.dtype == torch.float32
    assert output.pnp.geometry_weight.shape == (1, 5, 2)
    assert output.pnp.joint_weights.shape == (1, 5, 2, 21)


def test_all_disabled_path_has_identical_parameters_and_results():
    for arch in (CORE, FUSION):
        cfg = replace(tiny(), anchor_offset_cells=0)
        torch.manual_seed(21)
        before = HandPrismModel(ToyMano(), cfg, architecture=arch).eval()
        torch.manual_seed(21)
        explicit = HandPrismModel(ToyMano(), cfg, architecture=arch, fusion_config=FusionConfig()).eval()
        assert list(before.state_dict()) == list(explicit.state_dict())
        for key in before.state_dict():
            assert torch.equal(before.state_dict()[key], explicit.state_dict()[key])
        features = torch.randn(1, 2, 4, 4, 24)
        kwargs = dict(target_frames=5, solver="kfree", image_size=torch.tensor([[128., 128.]]))
        first, second = before(features, **kwargs), explicit(features, **kwargs)
        assert torch.equal(first.joints_camera, second.joints_camera)
    with pytest.raises(ValueError, match="Fusion refinements"):
        HandPrismModel(ToyMano(), replace(tiny(), anchor_offset_cells=0), architecture=CORE,
                       fusion_config=FusionConfig(final_readout=True))


def test_optional_modules_do_not_change_shared_initialization():
    torch.manual_seed(38)
    baseline = HandPrismModel(ToyMano(), replace(tiny(), anchor_offset_cells=0), architecture=FUSION)
    torch.manual_seed(38)
    enhanced = HandPrismModel(ToyMano(), tiny(), architecture=FUSION,
        fusion_config=FusionConfig(final_readout=True, local_rgb=True, joint_mano=True,
                                   temporal_wrist=True, reliability=True, edge_quality=True))
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, enhanced.state_dict()[key]), key


@pytest.mark.parametrize("frames", [1, 2, 5])
def test_empty_or_short_motion_intervals_have_finite_gradients(frames):
    from handprism.losses import masked_clip_mean
    pred = torch.randn(1, frames, 2, 3, requires_grad=True)
    truth = torch.randn_like(pred)
    mask = torch.ones(1, frames, 2, dtype=torch.bool)
    vel, vm, acc, am = motion_errors(pred, truth, mask, torch.arange(frames)[None].double()/30)
    loss = masked_clip_mean(vel, vm) + masked_clip_mean(acc, am)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_arctic_index_honors_valid_start_bounds_not_full_recording():
    from scripts.prepare_fusion_index import starts_for_record
    record = dict(start_min=10, start_max=100, num_frames=400)
    assert starts_for_record(record, 81, 81) == [10, 91, 100]
    dataset = HandPrismWindowDataset.__new__(HandPrismWindowDataset)
    dataset.training, dataset.frames, dataset.hard_window_fraction = True, 81, .5
    record["difficulty_windows"] = [dict(start_frame=200, strata=["edge"])]
    with pytest.raises(ValueError, match="outside valid"):
        for seed in range(100):
            dataset._start(record, seed)


def test_ablation_presets_are_independent_and_budgets_are_explicit():
    from scripts.make_v3_ablation import PRESETS, build_ablation
    configs = [load_config(ROOT/f"configs/handprism_fusion_{solver}.json") for solver in ("standard", "kfree")]
    for config in configs:
        for name, features in PRESETS.items():
            result = build_ablation(config, features, clip_budget=128000)
            assert result["ablation"]["global_clips"] == 128000
            assert result["dataset_weights"] == {"arctic": .4375, "hot3d": .5625}
            if name == "B0":
                assert not fusion_config_from_json(result).enabled
                assert loss_weights_from_json(result) == LossWeights()
            if name == "M2":
                assert result["fusion"]["local_rgb"] and not result["fusion"]["final_readout"]
        isolated = build_ablation(config, ("mano-loss",))
        assert not isolated["fusion"]["joint_mano"] and isolated["loss_weights"]["joints_root_mano"] > 0
    assert build_ablation(configs[0], (), clip_budget=128000)["steps"] == 2000
    assert build_ablation(configs[1], (), clip_budget=128000)["steps"] == 4000
    with pytest.raises(ValueError, match="requires an explicit mano-loss"):
        build_ablation(configs[0], ("consistency",))


def test_b0_readiness_requires_the_same_frozen_validation_index(tmp_path):
    from scripts.check_readiness import audit_config
    config = load_config(ROOT/"configs/handprism_fusion_b0_standard.json")
    config["manifests"] = str(tmp_path)
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    (tmp_path/"split_report.json").write_text(json.dumps({"fusion_index": {
        "version": 1, "test_used_for_design": False}}))
    for name in ("arctic", "hot3d"):
        (tmp_path/f"{name}_train.jsonl").write_text(json.dumps({"dataset": name, "difficulty_windows": []})+"\n")
        (tmp_path/f"{name}_val.jsonl").write_text('{}\n')
    checks = {}
    audit_config(ROOT, path, tmp_path, ("arctic", "hot3d"), checks, architecture=FUSION)
    assert not checks["fusion_difficulty_index_standard"]["pass"]
    from handprism.data.difficulty import FUSION_INDEX_VERSION, VALIDATION_WINDOW_POLICY
    (tmp_path/"split_report.json").write_text(json.dumps({"fusion_index": {
        "version": FUSION_INDEX_VERSION, "test_used_for_design": False,
        "val_windows_per_recording": 17,
        "validation_policy": VALIDATION_WINDOW_POLICY,
        "validation": {name: {"windows": 17, "recordings": 1} for name in ("arctic", "hot3d")}}}))
    for name in ("arctic", "hot3d"):
        (tmp_path/f"{name}_val.jsonl").write_text("".join(json.dumps({"dataset": name,
            "validation_strata": ["edge"], "recording_id": "val", "start_frame": i*81,
            "frames": 81, "num_frames": 81*17, "valid_ranges": [[0, 81*17]]})+"\n" for i in range(17)))
    audit_config(ROOT, path, tmp_path, ("arctic", "hot3d"), checks, architecture=FUSION)
    assert checks["fusion_difficulty_index_standard"]["pass"]


def test_nested_snapshot_does_not_claim_parent_git_identity():
    from scripts.train import git_fingerprint
    result = git_fingerprint(ROOT/"src")
    assert result["scope"] == "unversioned-source-snapshot"
    assert result["commit"] is None and result["dirty"] is None


def test_extended_contract_rejects_malformed_detail_and_unknown_visibility():
    from handprism.data.contract import validate_sample
    defaults = dict(video=torch.zeros(3, 5, 16, 16), timestamps=None, rgb_high=None,
        in_frame=None, observed=None, observed_valid=None, synthetic_occluded=None, visibility_valid=None)
    for invalid, message in (
        ({"rgb_high": torch.zeros(3, 5, dtype=torch.uint8)}, "rgb_high"),
        ({"observed": torch.zeros(5, 2, 21, dtype=torch.bool)}, "observed_valid"),
        ({"visibility_valid": torch.zeros(5, 2)}, "visibility_valid"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_sample(SimpleNamespace(**{**defaults, **invalid}))


def test_signed_loss_audit_and_nonfinite_guard():
    from handprism.fusion_runtime import weighted_loss_audit
    config = load_config(ROOT/"configs/handprism_fusion_kfree.json")
    terms = {"wrist_prior": -2., "camera_fit": .4, "joints_root": .1, "direct_mano_consistency": 0.}
    terms["total"] = -.2 + 5 * .5 * .4 + 5 * .1
    result = weighted_loss_audit(terms, config, 250)
    assert result["sum"] == pytest.approx(terms["total"])
    assert result["weighted"]["wrist_prior"] < 0
    with pytest.raises(RuntimeError, match="non-finite"):
        weighted_loss_audit({**terms, "camera_fit": float("nan")}, config, 250)


def test_oos_duration_censors_clip_boundaries_and_splits_gaps():
    from handprism.evaluator import oos_duration_masks
    mask = torch.ones(1, 81, 2, dtype=torch.bool)
    times = torch.arange(81)[None].double()/30
    masks, runs, censored = oos_duration_masks(mask, times, mask)
    assert runs == censored == 2
    assert masks["oos_within_clip_gt1s"].all()
    times[:, 41:] += 2
    _, runs, censored = oos_duration_masks(mask, times, mask)
    assert runs == censored == 4


def test_final_readout_uses_post_temporal_query():
    torch.manual_seed(8)
    decoder = HandPrismDecoder(tiny(), architecture=FUSION, fusion_config=FusionConfig(final_readout=True)).eval()
    features = torch.randn(1, 2, 4, 4, 24)
    ray = torch.randn(1, 4, 4, 3)
    before = decoder(features, ray, 5).anchors_2d
    with torch.no_grad():
        decoder.layers[-1].ffn.net[-1].bias[0] += 4
    after = decoder(features, ray, 5).anchors_2d
    assert not torch.allclose(before, after)


def test_roi_mapping_pixel_centers_batch_frames_and_invalid_fallback():
    class Mean(nn.Module):
        def forward(self, x):
            return x.mean((-1, -2))

    local = LocalRGB(3, FusionConfig(local_rgb=True, local_resolution=16, local_chunk_size=3)).eval()
    local.encoder, local.residual = Mean(), nn.Identity()
    # A coordinate ramp whose center is known analytically in source pixels.
    image = torch.arange(64).float()[None, :].expand(64, -1)[None].expand(3, -1, -1)
    rgb = image[None, :, None].expand(2, 3, 2, 64, 64).clone()
    rgb[1] += 100
    uv = torch.tensor([.25, .5]).expand(2, 2, 2, 21, 2).clone()
    hand = torch.zeros(2, 2, 2, 3)
    output, good, _ = local(rgb, uv, hand, image_size=torch.tensor([[32., 32.], [32., 32.]]))
    # global u=8 maps to (8+.5)*2-.5 = source pixel 16.5.
    torch.testing.assert_close(output[0, ..., :3], torch.full((2, 2, 3), 16.5))
    torch.testing.assert_close(output[1, ..., :3], torch.full((2, 2, 3), 116.5))
    assert good.all()
    uv.fill_(-4)
    output, good, _ = local(rgb, uv, hand, image_size=torch.tensor([[32., 32.], [32., 32.]]))
    assert not good.any() and torch.equal(output, torch.zeros_like(output))
    with pytest.raises(ValueError, match="GT ROIs"):
        local(rgb, uv, hand, uv, torch.ones(2, 2, 2, 21, dtype=torch.bool), image_size=torch.ones(2, 2)*32)


def test_native_mano_finger_mapping():
    module = FusionRefinement(16, FusionConfig(joint_mano=True))
    seen = []
    hook = module.joint_pose.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].detach()))
    decoder = HandPrismDecoder(tiny(), architecture=FUSION)
    out = decoder(torch.randn(1, 2, 4, 4, 24), torch.randn(1, 4, 4, 3), 5)
    out.joint_features[:] = torch.arange(21)[None, None, None, :, None]
    module(out, None)
    assert seen[0][0, 0, 0, :, 0].tolist() == [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
    hook.remove()


def test_timestamp_errors_handle_irregular_gaps_duplicates_and_fast_motion():
    times = torch.tensor([[1700000000., 1700000000.03, 1700000000.08, 1700000000.08,
                           1700000000.8, 1700000000.84]], dtype=torch.float64)
    relative = (times-times[:, :1]).float()
    truth = torch.stack((relative*2, relative.square()*3, relative*0), -1)[:, :, None]
    valid = torch.ones(1, 6, 1, dtype=torch.bool)
    vel, vm, acc, am = motion_errors(truth, truth, valid, times)
    assert (vel[vm] == 0).all() and (acc[am] == 0).all()
    assert vm.flatten().tolist() == [True, True, False, False, True]
    wrong = torch.zeros_like(truth, requires_grad=True)
    vel, vm, acc, am = motion_errors(wrong, truth, valid, times)
    assert vel[vm].mean() > 2 and acc[am].mean() > 5
    (vel[vm].mean()+acc[am].mean()).backward()
    assert torch.isfinite(wrong.grad).all()


def test_oos_prior_is_independent_of_anchor_zero_and_geometry_weights_are_detached():
    batch, _ = batch_data()
    joints, k, size = batch["joints_root"], batch["intrinsics"], batch["image_size"]
    anchors = torch.full((1, 5, 2, 21, 2), -1.)
    bearings = torch.randn_like(anchors)
    prior = torch.tensor([3., -.5, .7]).expand(1, 5, 2, 3).clone().requires_grad_()
    quality = torch.full(anchors.shape[:-1], .8, requires_grad=True)
    project = lambda p: (project_pinhole(p, k, size), p[..., 2] > .01)
    result = mixed_pnp(joints, anchors, torch.zeros(1, 5, 2, 1), bearings, size,
        projector=project, wrist_prior=prior, wrist_log_scale=torch.zeros_like(prior), joint_quality=quality)
    assert not result.solved.any()
    torch.testing.assert_close(result.translation, prior)
    result.translation.sum().backward()
    assert prior.grad.abs().sum() > 0 and quality.grad is None


@pytest.mark.parametrize("fraction", [0., .2])
def test_robust_geometry_stays_finite_and_depth_bounded(fraction):
    batch, _ = batch_data()
    joints, k, size = batch["joints_root"], batch["intrinsics"], batch["image_size"]
    anchors = batch["joints_2d"].clone()
    anchors[..., -2:, :] += .2
    bearings = bearings_from_intrinsics(anchors, k, size)
    log_depth = torch.full((1, 5, 2, 1), np.log(.7), requires_grad=True)
    result = mixed_pnp(joints, anchors, log_depth, bearings, size,
        SolverConfig(robust_iterations=2, depth_refine_fraction=fraction),
        projector=lambda p: (project_pinhole(p, k, size), p[..., 2] > .01))
    assert torch.isfinite(result.translation).all()
    assert ((result.translation[..., 2]-.7).abs() <= .7*fraction+1e-5).all()
    result.translation.square().sum().backward()
    assert torch.isfinite(log_depth.grad).all()


def test_metrics_are_solver_independent_and_oos_absolute_error_not_hidden():
    batch, vertices = batch_data()
    # Deliberately move every GT hand out of frame, retaining valid 3D.
    batch["translation"][..., 0] = 2.
    batch["joints_camera"] = batch["joints_root"] + batch["translation"][..., None, :]
    batch["joints_2d"] = project_pinhole(batch["joints_camera"], batch["intrinsics"], batch["image_size"])
    batch["valid_joints_2d"] = torch.zeros_like(batch["valid_joints_2d"])
    prediction = batch["joints_camera"].clone()
    prediction[..., 0] += .3
    output = SimpleNamespace(joints_camera=prediction, joints_root_mano=batch["joints_root"],
        vertices_root=vertices, vertices_camera=vertices+prediction[..., :1, :],
        pnp=SimpleNamespace(translation=prediction[..., 0, :]), mano_translation=prediction[..., 0, :],
        decoder=SimpleNamespace(anchors_2d=batch["joints_2d"], global_rotation=batch["global_rotation"],
                                existence_logits=torch.ones(1, 5, 2)*-10))
    results = []
    for solver in ("standard", "kfree"):
        accumulator = EvaluationAccumulator()
        score_batch(accumulator, output, batch, vertices, batch["joints_root"][0, 0],
                    solver=solver, gt_mano_translation=batch["translation"])
        result = accumulator.finalize()
        validate_metric_result(result)
        results.append(result)
        assert result["MPJPE-OOS_mm"] == 0
        assert result["oos/WristAbsolute_mm"] == pytest.approx(300, abs=.001)
        assert result["oos/ExistenceCoverage"] == 0
        assert result["occluded/count"] == 0
    assert results[0] == results[1]


def test_quality_risk_and_prior_coverage_do_not_hide_undetected_hands():
    batch, vertices = batch_data()
    anchors = batch["joints_2d"].clone()
    anchors[..., 0] += 8 / 128
    quality = torch.full((1, 5, 2, 21), .1)
    quality[..., :10] = .9
    prior = batch["translation"].clone()
    prior[..., 0] += .2
    output = SimpleNamespace(joints_camera=batch["joints_camera"], joints_root_mano=batch["joints_root"],
        vertices_root=vertices, vertices_camera=vertices+batch["translation"][..., None, :],
        pnp=SimpleNamespace(translation=batch["translation"]), mano_translation=batch["translation"],
        decoder=SimpleNamespace(anchors_2d=anchors, global_rotation=batch["global_rotation"],
            existence_logits=torch.full((1, 5, 2), -10.), reliability_logits=quality.logit(),
            wrist_prior=prior, wrist_log_scale=torch.full_like(prior, np.log(.005))))
    accumulator = EvaluationAccumulator()
    score_batch(accumulator, output, batch, vertices, batch["joints_root"][0, 0],
                solver="standard", gt_mano_translation=batch["translation"])
    result = accumulator.finalize()
    assert result["ExistenceCoverage"] == 0
    assert result["quality_at_0.5/coverage"] == pytest.approx(10 / 21)
    assert result["quality_at_0.5/AnchorEPE_px"] == pytest.approx(8, abs=1e-5)
    assert result["QualitySoftECE"] == pytest.approx(float((quality-np.exp(-1)).abs().mean()))
    assert result["WristPrior95AxisCoverage"] == pytest.approx(2 / 3)
    assert result["WristPriorConfidentFailureRate"] == 1
    merged = EvaluationAccumulator()
    merged.load_tensor(accumulator.as_tensor(torch.device("cpu")) * 2)
    assert merged.finalize()["QualitySoftECE"] == result["QualitySoftECE"]


def test_appearance_augmentation_is_coherent_and_does_not_change_geometry():
    video = torch.linspace(-1, 1, 3*32*32).reshape(3, 1, 32, 32).expand(-1, 5, -1, -1).clone()
    sample = SimpleNamespace(video=video, rgb_high=None, joints_2d=torch.ones(5, 2, 21, 2)*.5,
                             valid_joints_2d=torch.ones(5, 2, 21, dtype=torch.bool))
    config = AugmentationConfig(enabled=True, jpeg_probability=1., blur_probability=1.)
    before = copy.deepcopy(sample)
    first = augment_sample(copy.deepcopy(sample), config, 41)
    second = augment_sample(copy.deepcopy(sample), config, 41)
    assert torch.equal(first.video, second.video)
    assert torch.equal(first.video[:, 0], first.video[:, -1])
    assert torch.equal(first.joints_2d, before.joints_2d)
    assert not torch.equal(first.video, before.video)


def test_difficulty_sampling_is_deterministic_and_stays_in_valid_ranges():
    dataset = HandPrismWindowDataset.__new__(HandPrismWindowDataset)
    dataset.training, dataset.frames, dataset.hard_window_fraction = True, 81, .5
    record = dict(num_frames=300, valid_ranges=[[0, 100], [150, 300]],
                  difficulty_windows=[dict(start_frame=160, strata=["oos", "edge"])])
    starts = [dataset._start(record, seed) for seed in range(100)]
    assert starts == [dataset._start(record, seed) for seed in range(100)]
    assert starts.count(160) > 25 and len(set(starts)) > 15
    assert all(0 <= start <= 19 or 150 <= start <= 219 for start in starts)
    records = [dict(recording_id=str(i), split_group=i%3, validation_strata=["oos"] if i%4==0 else []) for i in range(19)]
    order = stratified_order(records)
    assert sorted(order) == list(range(19)) and order == stratified_order(records)


def test_selection_uses_accuracy_coverage_not_loss_and_requires_both_datasets():
    metrics = {}
    for dataset in ("arctic", "hot3d"):
        for key, value in dict(CameraMPJPE_mm=50, **{"MPJPE+OOS_mm":20}, AnchorEPE_px=10, ExistenceCoverage=.9, F1=.8).items():
            metrics[f"val/{dataset}/test_protocol/{key}"] = value
    score = validation_selection_score(metrics)
    metrics["val/mean_loss"] = -1000
    assert validation_selection_score(metrics) == score
    metrics["val/arctic/test_protocol/ExistenceCoverage"] = .1
    assert validation_selection_score(metrics) > score
    del metrics["val/hot3d/test_protocol/F1"]
    with pytest.raises(ValueError):
        validation_selection_score(metrics)


def test_r3_rejects_old_checkpoints_unknown_options_and_missing_rgb(tmp_path):
    config = load_config(ROOT/"configs/handprism_fusion_standard.json")
    with pytest.raises(ValueError):
        validate_checkpoint_identity(dict(format=CHECKPOINT_FORMAT, architecture=FUSION,
            implementation_id="handprism-fusion-r2", config=config), config, architecture=FUSION)
    with pytest.raises(ValueError, match="unknown Fusion"):
        FusionConfig.from_dict({"typo": True})
    with pytest.raises(ValueError, match="original-resolution"):
        fusion_forward_options(config, {})
    clip = tmp_path/"clip.npz"
    np.savez(clip, video=np.zeros((5, 32, 32, 3), np.uint8), rgb_high=np.zeros((5, 64, 64, 3), np.uint8),
             intrinsics=np.array([float("nan")]), timestamps=np.arange(5)/30)
    batch = load_clip(clip, "kfree")
    assert batch["intrinsics"] is None and batch["rgb_high"].dtype == torch.uint8
    assert batch["timestamps"].dtype == torch.float64
