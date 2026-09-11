"""Regression cases from the September 6 audit; no production dataset I/O."""

import json
import math
from pathlib import Path

import pytest
import torch

from handprism.camera import project_pinhole
from handprism.config import DecoderConfig, SolverConfig
from handprism.decoder import HandPrismDecoder, interpolate_time
from handprism.data.dataset import read_jsonl
from handprism.data.policy import manifest_path
from handprism.mano import ToyMano
from handprism.model import HandPrismModel
from handprism.ray import (
    fit_effective_pinhole_camera,
    mixed_pnp,
    normalized_pixel_grid,
    project_via_ray_field,
    sample_ray_bearings,
)
from handprism.losses import camera_fit_bearing_loss, HandPrismLoss
from handprism.completion import validated_metrics, file_sha256
from handprism.version import IMPLEMENTATION_ID
from handprism.architectures import FUSION
from scripts.run_pipeline import evaluation_directory, resource_preflight
from scripts.train import load_config, restore_checkpoint, validate


def tiny_config():
    return DecoderConfig(feature_dim=24, hidden_dim=16, layers=2, heads=4, ffn_dim=32)


def decoder_input():
    torch.manual_seed(43)
    features = torch.randn(1, 3, 4, 4, 24)
    rays = torch.tensor([0.0, 0.0, 1.0]).expand(1, 4, 4, 3)
    return features, rays


def test_registers_influence_readout_and_receive_gradients():
    features, rays = decoder_input()
    model = HandPrismDecoder(tiny_config(), architecture="handprism-fusion").eval()
    before = model(features, rays, 7)
    before.global_rotation_6d.square().sum().backward()
    assert model.queries.grad[-4:].abs().sum() > 1e-6
    with torch.no_grad():
        model.queries[-4:] += torch.randn_like(model.queries[-4:])
    after = model(features, rays, 7)
    assert (before.global_rotation_6d - after.global_rotation_6d).abs().max() > 1e-5


def test_joint_queries_can_communicate_with_hand_queries():
    features, rays = decoder_input()
    model = HandPrismDecoder(tiny_config(), architecture="handprism-fusion").eval()
    before = model(features, rays, 7).global_rotation_6d
    with torch.no_grad():
        model.queries[2:44] += torch.randn_like(model.queries[2:44])
    after = model(features, rays, 7).global_rotation_6d
    assert (before - after).abs().max() > 1e-5


def test_direct_joint_coordinates_are_interpolated_after_latent_mlp():
    features, rays = decoder_input()
    model = HandPrismDecoder(tiny_config(), architecture="handprism-fusion").eval()
    output = model(features, rays, 7)
    joint_tokens = output.query_features_latent[:, :, 2:44].reshape(1, 3, 2, 21, 16)
    points = model.joint_head(joint_tokens)
    points = points - points[..., :1, :]
    torch.testing.assert_close(output.joints_root_direct, interpolate_time(points, 7))


def test_pixel_gate_does_not_scale_bearings_by_image_diagonal():
    joints = torch.zeros(1, 1, 2, 21, 3, requires_grad=True)
    bearings = torch.zeros(1, 1, 2, 21, 2)
    bearings[..., 1::2, 0] = 0.05
    bearings[..., 2::2, 0] = -0.05
    intrinsics = torch.tensor([[[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]])
    size = torch.tensor([[480.0, 640.0]])
    anchors = (bearings * 200 + torch.tensor([320.0, 240.0])) / torch.tensor([640.0, 480.0])
    output = mixed_pnp(
        joints,
        anchors,
        torch.zeros(1, 1, 2, 1),
        bearings,
        size,
        projector=lambda points: (project_pinhole(points, intrinsics, size), points[..., 2] > 0),
    )
    assert output.solved.all()  # Old proxy reports ~39 px and rejects this solve.
    torch.testing.assert_close(output.rms_pixels, torch.full((1, 1, 2), 10 * math.sqrt(20 / 21)))
    output.translation.sum().backward()
    assert torch.isfinite(joints.grad).all()


def fish_rays():
    grid = normalized_pixel_grid(32, 32, device=torch.device("cpu"), dtype=torch.float32)
    offset = grid - 0.5
    radius = offset.norm(dim=-1, keepdim=True)
    theta = radius / 0.6
    return torch.cat((offset / radius.clamp_min(1e-8) * theta.sin(), theta.cos()), -1)[None]


def test_nonpinhole_field_rejects_fit_and_has_accurate_pixel_inverse():
    rays = fish_rays()
    camera = fit_effective_pinhole_camera(rays)
    assert camera.numerical_valid.all()
    assert not camera.valid.any()
    assert camera.rms_normalized.min() > 0.01
    uv = torch.tensor([[[[[0.2, 0.25], [0.7, 0.8], [0.5, 0.5]]]]])
    bearings = sample_ray_bearings(rays, uv)
    points = torch.cat((bearings, torch.ones_like(bearings[..., :1])), -1)
    projected, valid = project_via_ray_field(points, rays, uv + 0.02)
    assert valid.all()
    torch.testing.assert_close(projected, uv, atol=1e-4, rtol=1e-4)


def test_constant_ray_inverse_fails_closed_without_nonfinite_diagnostics():
    rays = torch.tensor([0.0, 0.0, 1.0]).expand(1, 8, 8, 3)
    uv = torch.full((1, 1, 2, 21, 2), 0.5)
    points = torch.tensor([0.0, 0.0, 1.0]).expand(1, 1, 2, 21, 3)
    projected, valid = project_via_ray_field(points, rays, uv)
    assert not valid.any()
    assert torch.isfinite(projected).all()


def test_ray_inverse_rejects_ambiguous_border_padding():
    grid = normalized_pixel_grid(8, 8, device=torch.device("cpu"), dtype=torch.float32)
    rays = torch.nn.functional.normalize(torch.cat((grid - 0.5, torch.ones(8, 8, 1)), -1), dim=-1)[
        None
    ]
    uv = torch.tensor([[[[[0.5, 0.98]]]]])
    bearings = sample_ray_bearings(rays, uv)
    points = torch.cat((bearings, torch.ones_like(bearings[..., :1])), -1)
    _, valid = project_via_ray_field(points, rays, uv)
    assert not valid.any()  # Multiple UVs share one border ray; do not fake an inverse.


def test_fit_loss_does_not_force_a_fisheye_target_to_pinhole():
    grid = normalized_pixel_grid(32, 32, device=torch.device("cpu"), dtype=torch.float32)
    rays = torch.nn.functional.normalize(
        torch.cat(((grid - 0.5) / 0.8, torch.ones(32, 32, 1)), -1), dim=-1
    )[None].requires_grad_()
    loss = camera_fit_bearing_loss(fit_effective_pinhole_camera(rays), fish_rays(), None, "l1")
    assert loss == 0
    loss.backward()
    assert torch.isfinite(rays.grad).all()


@pytest.mark.parametrize("architecture", ["handprism-core", "handprism-fusion"])
def test_kfree_forward_ignores_all_ground_truth_camera_inputs(architecture):
    features, _ = decoder_input()
    model = HandPrismModel(ToyMano(), tiny_config(), architecture=architecture).eval()
    size = torch.tensor([[480.0, 640.0]])
    first = model(features, target_frames=7, solver="kfree", image_size=size)
    second = model(
        features,
        target_frames=7,
        solver="kfree",
        image_size=size,
        intrinsics=torch.randn(1, 3, 3),
        distortion=torch.randn(1, 8),
        calibration_ray_field=torch.randn(1, 4, 4, 3),
        camera_model="unknown",
        camera_parameters=torch.randn(1, 15),
        source_image_size=torch.ones(1, 2),
    )
    torch.testing.assert_close(first.pnp.translation, second.pnp.translation)
    torch.testing.assert_close(first.pnp.solved, second.pnp.solved)


def test_forbidden_manifest_is_rejected_before_open(monkeypatch):
    def no_open(*args, **kwargs):
        pytest.fail("a forbidden manifest must never be opened")

    monkeypatch.setattr(Path, "open", no_open)
    with pytest.raises(ValueError):
        read_jsonl("egodex_train.jsonl")


def test_mislabeled_manifest_record_rejected_without_dataset_io(tmp_path):
    path = tmp_path / "arctic_train.jsonl"
    path.write_text(json.dumps({"dataset": "other", "split": "train"}) + "\n")
    with pytest.raises(ValueError, match="unsupported dataset"):
        read_jsonl(path)


def test_old_config_cannot_start_v3(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs/two_dataset_v3.json").read_text())
    config.pop("implementation_id")
    path = tmp_path / "old.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="implementation_id"):
        load_config(path)


def test_old_checkpoint_is_rejected_before_loading_parameters(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format": "unsupported-training-checkpoint"}, path)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/handprism_fusion_standard.json")
    with pytest.raises(ValueError, match="checkpoint architecture/implementation mismatch"):
        restore_checkpoint(path, None, None, None, 0, 1, config)


def test_validation_reports_direct_mano_and_final_test_protocol():
    from handprism.completion import require_finite_json

    class SmallSystem(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hand = HandPrismModel(ToyMano(), tiny_config(), architecture="handprism-fusion")

        def forward(self, features, **kwargs):
            return self.hand(features, **kwargs)

    features, _ = decoder_input()
    system = SmallSystem()
    rotation = torch.eye(3).expand(1, 3, 2, 3, 3)
    articulation = torch.eye(3).expand(1, 3, 2, 15, 3, 3)
    betas = torch.zeros(1, 3, 2, 10)
    joints, _ = system.hand.mano(rotation, articulation, betas)
    translation = torch.tensor([0.0, 0.0, 0.8]).expand(1, 3, 2, 3)
    camera_points = joints + translation.unsqueeze(-2)
    k = torch.tensor([[[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]])
    size = torch.tensor([[480.0, 640.0]])
    hands = torch.ones(1, 3, 2, dtype=torch.bool)
    valid_joints = torch.ones(1, 3, 2, 21, dtype=torch.bool)
    batch = {
        "dataset": "arctic",
        "video": torch.zeros(1, 3, 3, 8, 8),
        "global_rotation": rotation,
        "articulation": articulation,
        "betas": betas,
        "joints_root": joints,
        "joints_camera": camera_points,
        "translation": translation,
        "joints_2d": project_pinhole(camera_points, k, size),
        "intrinsics": k,
        "image_size": size,
        "distortion": torch.zeros(1, 8),
        "camera_model": "pinhole",
        "camera_parameters": None,
        "source_image_size": None,
        "gt_ray_field": None,
        "valid_hand": hands,
        "valid_mano": hands,
        "existence": hands.float(),
        "visibility": hands.float(),
        "valid_joints_3d": valid_joints,
        "valid_joints_2d": valid_joints,
        "valid_ray": torch.ones(1, dtype=torch.bool),
    }
    config = {"solver": "standard", "steps": 500, "validation_batches_per_dataset": 1,
              "architecture": "handprism-fusion", "fusion": {}}
    result = validate(
        system,
        lambda video: features,
        HandPrismLoss(),
        {"arctic": [batch]},
        config,
        torch.device("cpu"),
        1,
        torch.float32,
    )
    require_finite_json(result)
    for key in (
        "direct_root_mpjpe_mm",
        "mano_root_mpjpe_mm",
        "anchors_epe_px",
        "test_protocol/MPJPE-p_mm",
        "test_protocol/F1",
        "projection_failure_fraction",
    ):
        assert f"val/arctic/{key}" in result
    assert "val/arctic/root_mpjpe_mm" not in result


def completed_report(checkpoint):
    return {
        "architecture": FUSION,
        "implementation_id": IMPLEMENTATION_ID,
        "solver": "standard",
        "checkpoint_step": 20000,
        "full_test": True,
        "checkpoint_sha256": file_sha256(checkpoint),
        "datasets": {"arctic": {"segments": 291}, "hot3d": {"segments": 437}},
        "overall": {"segments": 728},
    }


@pytest.mark.parametrize("fault", ["json", "nan", "subset", "step", "third", "hash"])
def test_invalid_metrics_never_complete_a_run(tmp_path, fault):
    checkpoint = tmp_path / "step_020000.pt"
    checkpoint.write_bytes(b"unit-test-checkpoint")
    directory = tmp_path / "evaluation_step_020000"
    directory.mkdir()
    report = completed_report(checkpoint)
    if fault == "nan":
        report["overall"]["loss"] = float("nan")
    if fault == "subset":
        report["datasets"]["hot3d"]["segments"] = 1
    if fault == "step":
        report["checkpoint_step"] = 500
    if fault == "third":
        report["datasets"]["other"] = {"segments": 1}
    if fault == "hash":
        report["checkpoint_sha256"] = "wrong"
    (directory / "metrics.json").write_text("{" if fault == "json" else json.dumps(report))
    with pytest.raises(ValueError):
        validated_metrics(
            directory,
            architecture=FUSION,
            solver="standard",
            step=20000,
            checkpoint=checkpoint,
            expected_counts={"arctic": 291, "hot3d": 437},
        )
    selected = evaluation_directory(
        tmp_path,
        20000,
        architecture=FUSION,
        solver="standard",
        checkpoint=checkpoint,
        expected_counts={"arctic": 291, "hot3d": 437},
    )
    assert selected.name.endswith("_attempt_2")


def test_valid_completed_attempt_is_reused(tmp_path):
    checkpoint = tmp_path / "step_020000.pt"
    checkpoint.write_bytes(b"unit-test-checkpoint")
    (tmp_path / "evaluation_step_020000").mkdir()
    directory = tmp_path / "evaluation_step_020000_attempt_2"
    directory.mkdir()
    (directory / "metrics.json").write_text(json.dumps(completed_report(checkpoint)))
    assert (
        evaluation_directory(
            tmp_path,
            20000,
            architecture=FUSION,
            solver="standard",
            checkpoint=checkpoint,
            expected_counts={"arctic": 291, "hot3d": 437},
        )
        == directory
    )


def test_resource_preflight_refuses_insufficient_disk(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "scripts.run_pipeline.shutil.disk_usage",
        lambda _: SimpleNamespace(free=170 * 1024**3),
    )
    with pytest.raises(RuntimeError, match="insufficient disk"):
        resource_preflight(
            tmp_path,
            (("standard", None, tmp_path / "standard"), ("kfree", None, tmp_path / "kfree")),
        )


def test_resource_preflight_refuses_busy_gpus(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "scripts.run_pipeline.shutil.disk_usage",
        lambda _: SimpleNamespace(free=300 * 1024**3),
    )
    monkeypatch.setattr(
        "scripts.run_pipeline.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="1234\n"),
    )
    with pytest.raises(RuntimeError, match="compute jobs"):
        resource_preflight(tmp_path, (("standard", None, tmp_path / "standard"),))
