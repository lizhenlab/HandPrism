from __future__ import annotations

import math

import torch

from dreamhand.camera import project_fisheye624_upright, project_pinhole
from dreamhand.config import DecoderConfig, DreamHandConfig, SolverConfig
from dreamhand.decoder import DreamHandDecoder
from dreamhand.mano import ToyMano
from dreamhand.metrics import jitter_matched_runs, procrustes_mpjpe
from dreamhand.model import DreamHandModel
from dreamhand.losses import (
    DreamHandLoss,
    DreamHandPrediction,
    DreamHandTarget,
    camera_fit_bearing_loss,
    camera_fit_warmup_factor,
    masked_mean,
)
from dreamhand.positional import RayPE
from dreamhand.ray import (
    RayHead,
    kfree_bearings,
    bearings_from_effective_camera,
    distort_bearings,
    fit_effective_pinhole_camera,
    mixed_pnp,
    normalized_pixel_grid,
    undistort_bearings,
)
from dreamhand.rotations import geodesic_distance, rotation_6d_to_matrix


def test_component_config_contract() -> None:
    config = DreamHandConfig()
    config.validate()
    assert config.decoder.queries == 48
    assert config.latent_frames == 21


def test_rotation_6d_is_orthonormal() -> None:
    value = torch.randn(7, 6)
    rotation = rotation_6d_to_matrix(value)
    identity = rotation.transpose(-1, -2) @ rotation
    torch.testing.assert_close(identity, torch.eye(3).expand_as(identity), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.det(rotation), torch.ones(7), atol=1e-5, rtol=1e-5)


def test_geodesic_identity_is_exactly_zero() -> None:
    identity = torch.eye(3).expand(4, 3, 3)
    torch.testing.assert_close(geodesic_distance(identity, identity), torch.zeros(4))


def test_ray_head_default_parameter_count_and_zero_start() -> None:
    head = RayHead()
    assert sum(parameter.numel() for parameter in head.parameters()) == 9_219
    output = head(torch.randn(1, 2, 3, 4, 3072))
    assert output.shape == (1, 3, 4, 3)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[..., :2], torch.zeros_like(output[..., :2]))
    torch.testing.assert_close(output[..., 2], torch.ones_like(output[..., 2]))


def synthetic_pinhole_ray_field(
    focal: torch.Tensor,
    principal: torch.Tensor,
    height: int = 8,
    width: int = 10,
) -> torch.Tensor:
    pixels = normalized_pixel_grid(
        height,
        width,
        device=focal.device,
        dtype=focal.dtype,
    )
    bearings = (pixels.unsqueeze(0) - principal[:, None, None]) / focal[:, None, None]
    return torch.nn.functional.normalize(
        torch.cat((bearings, torch.ones_like(bearings[..., :1])), dim=-1),
        dim=-1,
    )


def test_kfree_effective_camera_fit_recovers_pinhole() -> None:
    focal = torch.tensor([[0.72, 0.88]])
    principal = torch.tensor([[0.47, 0.53]])
    ray_field = synthetic_pinhole_ray_field(focal, principal)
    camera = fit_effective_pinhole_camera(ray_field)
    assert camera.valid.all()
    torch.testing.assert_close(camera.focal, focal, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(camera.principal, principal, atol=1e-5, rtol=1e-5)
    anchors = torch.rand(1, 3, 2, 21, 2)
    expected = (anchors - principal.view(1, 1, 1, 1, 2)) / focal.view(1, 1, 1, 1, 2)
    torch.testing.assert_close(
        bearings_from_effective_camera(anchors, camera),
        expected,
        atol=2e-5,
        rtol=2e-5,
    )


def test_kfree_failed_fit_falls_back_to_direct_ray_sampling() -> None:
    ray_field = torch.tensor([0.0, 0.0, 1.0]).expand(1, 4, 5, 3).clone()
    anchors = torch.rand(1, 2, 2, 21, 2)
    bearings, camera = kfree_bearings(ray_field, anchors)
    assert not camera.valid.any()
    torch.testing.assert_close(bearings, torch.zeros_like(bearings))


def test_kfree_camera_fit_loss_gradient_and_warmup() -> None:
    predicted = synthetic_pinhole_ray_field(
        torch.tensor([[0.70, 0.82]]),
        torch.tensor([[0.49, 0.51]]),
    ).requires_grad_()
    target = synthetic_pinhole_ray_field(
        torch.tensor([[0.78, 0.91]]),
        torch.tensor([[0.46, 0.55]]),
    )
    camera = fit_effective_pinhole_camera(predicted)
    loss = camera_fit_bearing_loss(
        camera,
        target,
        torch.ones(target.shape[:-1], dtype=torch.bool),
        "l1",
    )
    assert loss > 0
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0
    assert camera_fit_warmup_factor(0, 500) == 0.0
    assert camera_fit_warmup_factor(250, 500) == 0.5
    assert camera_fit_warmup_factor(500, 500) == 1.0
    assert camera_fit_warmup_factor(1000, 500) == 1.0


def test_ray_pe_zero_output_has_live_data_dependent_gradient() -> None:
    ray_pe = RayPE(width=32, frequencies=2)
    rays = torch.randn(2, 3, 4, 3)
    output = ray_pe(rays)
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    first = ray_pe.net[0]
    final = ray_pe.net[-1]
    assert isinstance(first, torch.nn.Linear) and isinstance(final, torch.nn.Linear)
    assert first.weight.abs().sum() > 0
    assert final.weight.grad is not None and final.weight.grad.abs().sum() > 0


def test_mixed_pnp_recovers_synthetic_translation() -> None:
    generator = torch.Generator().manual_seed(1)
    joints = torch.randn(1, 2, 2, 21, 3, generator=generator) * 0.02
    joints[..., 0, :] = 0.0
    truth = torch.tensor([0.08, -0.04, 0.75]).view(1, 1, 1, 3).expand(1, 2, 2, 3)
    placed = joints + truth.unsqueeze(-2)
    bearings = placed[..., :2] / placed[..., 2:]
    intrinsics = torch.tensor([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    anchors = torch.stack(
        (
            (bearings[..., 0] * intrinsics[0, 0] + intrinsics[0, 2]) / 640.0,
            (bearings[..., 1] * intrinsics[1, 1] + intrinsics[1, 2]) / 480.0,
        ),
        dim=-1,
    )
    output = mixed_pnp(
        joints,
        anchors,
        torch.full((1, 2, 2, 1), math.log(0.75)),
        bearings,
        torch.tensor([[480.0, 640.0]]),
        SolverConfig(max_rms_px=1e6),
        projector=lambda points: (
            project_pinhole(points, intrinsics[None], torch.tensor([[480.0, 640.0]])),
            points[..., 2] > 0,
        ),
    )
    assert output.solved.all()
    torch.testing.assert_close(output.translation, truth, atol=1e-5, rtol=1e-5)


def test_mixed_pnp_wrist_fallback() -> None:
    joints = torch.zeros(1, 1, 2, 21, 3)
    anchors = torch.full((1, 1, 2, 21, 2), -1.0)
    bearings = torch.zeros_like(anchors)
    output = mixed_pnp(
        joints,
        anchors,
        torch.zeros(1, 1, 2, 1),
        bearings,
        torch.tensor([[480.0, 640.0]]),
        projector=lambda points: (torch.zeros_like(anchors), points[..., 2] > 0),
    )
    assert output.used_fallback.all()
    expected = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 1, 3).expand_as(output.translation)
    torch.testing.assert_close(output.translation, expected)


def test_distortion_round_trip() -> None:
    bearings = torch.tensor([[[0.15, -0.25], [0.40, 0.30]]])
    coefficients = torch.tensor([[0.12, -0.03, 0.001, -0.002, 0.01, 0.02, -0.01, 0.005]])
    distorted = distort_bearings(bearings, coefficients)
    recovered = undistort_bearings(distorted, coefficients, iterations=16)
    torch.testing.assert_close(recovered, bearings, atol=2e-5, rtol=2e-5)


def test_masked_mean_does_not_leak_invalid_nan() -> None:
    value = torch.tensor([float("nan"), 2.0])
    assert masked_mean(value, torch.tensor([False, True])) == 2.0


def test_joint_only_capability_does_not_train_mano_derived_losses() -> None:
    batch, frames, hands, joints = 1, 3, 2, 21
    identity = torch.eye(3).expand(batch, frames, hands, 3, 3).clone()
    articulation = torch.eye(3).expand(batch, frames, hands, 15, 3, 3).clone()
    camera = torch.arange(frames, dtype=torch.float32).square().view(1, frames, 1, 1, 1)
    camera = camera.expand(batch, frames, hands, joints, 3).clone()
    camera[..., 2] += 1.0
    ray = torch.tensor([0.0, 0.0, 1.0]).expand(batch, 2, 2, 3).clone()
    prediction = DreamHandPrediction(
        global_rotation=identity,
        articulation=articulation,
        betas=torch.zeros(batch, hands, 10),
        joints_root_direct=torch.ones(batch, frames, hands, joints, 3),
        joints_root_mano=torch.ones(batch, frames, hands, joints, 3),
        joints_camera=camera,
        translation=torch.ones(batch, frames, hands, 3),
        anchors_2d=torch.ones(batch, frames, hands, joints, 2),
        existence_logits=torch.zeros(batch, frames, hands),
        visibility_logits=torch.zeros(batch, frames, hands),
        ray_field=torch.tensor([1.0, 0.0, 0.0]).expand_as(ray).clone(),
    )
    target = DreamHandTarget(
        global_rotation=identity,
        articulation=articulation,
        betas=torch.zeros(batch, frames, hands, 10),
        joints_root=torch.zeros(batch, frames, hands, joints, 3),
        joints_camera=torch.zeros(batch, frames, hands, joints, 3),
        translation=torch.zeros(batch, frames, hands, 3),
        joints_2d=torch.zeros(batch, frames, hands, joints, 2),
        existence=torch.ones(batch, frames, hands, dtype=torch.bool),
        visibility=torch.ones(batch, frames, hands, dtype=torch.bool),
        ray_field=ray,
        valid_hand=torch.ones(batch, frames, hands, dtype=torch.bool),
        valid_mano=torch.zeros(batch, frames, hands, dtype=torch.bool),
        valid_joints_3d=torch.ones(batch, frames, hands, joints, dtype=torch.bool),
        valid_joints_2d=torch.zeros(batch, frames, hands, joints, dtype=torch.bool),
        valid_ray=torch.zeros(batch, 2, 2, dtype=torch.bool),
    )
    losses = DreamHandLoss()(
        prediction,
        target,
        torch.eye(3).unsqueeze(0),
        torch.tensor([[480.0, 640.0]]),
    )
    assert losses["joints_root"] > 0
    for name in (
        "rotation_geodesic",
        "rotation_matrix",
        "shape",
        "joints_camera",
        "wrist",
        "anchors_2d",
        "reprojection_joints",
        "reprojection_wrist",
        "translation",
        "acceleration",
        "ray",
        "camera_fit",
    ):
        assert losses[name] == 0


def test_fisheye624_upright_projection_is_differentiable() -> None:
    points = torch.tensor([[[[[0.0, 0.0, 1.0], [0.05, -0.10, 1.0]]]]], requires_grad=True)
    parameters = torch.tensor(
        [[600.0, 704.0, 704.0, 0.1, -0.02, 0.003, 0.0, 0.0, 0.0, 1e-3, -2e-3, 0.0, 0.0, 0.0, 0.0]]
    )
    image_size = torch.tensor([[480.0, 480.0]])
    projected = project_fisheye624_upright(
        points, parameters, torch.tensor([[1408.0, 1408.0]]), image_size
    )
    expected_center = torch.tensor(
        [(1408.0 - 0.5 - 704.0) / 1408.0 - 0.5 / 480.0, (704.0 + 0.5) / 1408.0 - 0.5 / 480.0]
    )
    torch.testing.assert_close(projected[0, 0, 0, 0], expected_center)
    projected.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def tiny_decoder_config() -> DecoderConfig:
    return DecoderConfig(
        feature_dim=32,
        hidden_dim=32,
        layers=2,
        heads=4,
        ffn_dim=64,
        spatial_pe_height=4,
        spatial_pe_width=4,
        ray_frequencies=2,
    )


def test_decoder_shapes_and_heatmap_normalization() -> None:
    decoder = DreamHandDecoder(tiny_decoder_config(), architecture="handprism-fusion")
    features = torch.randn(2, 3, 4, 5, 32)
    rays = torch.randn(2, 4, 5, 3)
    output = decoder(features, rays, target_frames=9)
    assert output.global_rotation.shape == (2, 9, 2, 3, 3)
    assert output.articulation.shape == (2, 9, 2, 15, 3, 3)
    assert output.betas.shape == (2, 2, 10)
    assert output.joints_root_direct.shape == (2, 9, 2, 21, 3)
    assert output.anchors_2d.shape == (2, 9, 2, 21, 2)
    total = output.attention_heatmaps.flatten(-2).sum(-1)
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-5, rtol=1e-5)
    assert ((output.anchors_2d >= 0) & (output.anchors_2d <= 1)).all()


def test_full_tiny_model_backward() -> None:
    model = DreamHandModel(ToyMano(), tiny_decoder_config(), architecture="handprism-fusion")
    features = torch.randn(1, 3, 4, 5, 32, requires_grad=True)
    output = model(
        features,
        target_frames=9,
        solver="standard",
        intrinsics=torch.tensor([[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]]),
        image_size=torch.tensor([[480.0, 640.0]]),
    )
    assert output.joints_camera.shape == (1, 9, 2, 21, 3)
    assert output.vertices_camera.shape == (1, 9, 2, 778, 3)
    loss = output.joints_camera.square().mean() + output.decoder.anchors_2d.square().mean()
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_full_tiny_kfree_model_exposes_camera_fit() -> None:
    model = DreamHandModel(ToyMano(), tiny_decoder_config(), architecture="handprism-fusion")
    output = model(
        torch.randn(1, 3, 4, 5, 32),
        target_frames=5,
        solver="kfree",
        image_size=torch.tensor([[480.0, 640.0]]),
    )
    assert output.camera_fit is not None
    assert output.camera_fit.focal.shape == (1, 2)
    # The zero-initialized ray head is an optical-axis field, so the configured
    # variance guard must select direct sampling until the ray head learns.
    assert not output.camera_fit.valid.any()
    assert torch.isfinite(output.joints_camera).all()


def test_procrustes_removes_similarity_transform() -> None:
    target = torch.randn(2, 21, 3)
    angle = torch.tensor(0.4)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    prediction = 1.7 * (target @ rotation) + torch.tensor([0.2, -0.3, 1.1])
    assert procrustes_mpjpe(prediction, target).max() < 1e-5


def test_jitter_respects_detection_run_boundaries() -> None:
    joints = torch.zeros(8, 21, 3)
    joints[1, :, 0] = 1
    joints[2, :, 0] = 2
    joints[5, :, 0] = 10
    joints[6, :, 0] = 11
    joints[7, :, 0] = 12
    matched = torch.tensor([True, True, True, False, False, True, True, True])
    torch.testing.assert_close(jitter_matched_runs(joints, matched), torch.tensor(0.0))
