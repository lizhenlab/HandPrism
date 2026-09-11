"""HandPrism multi-task losses with per-clip reduction and capability masks."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .camera import PINHOLE, project_camera, project_pinhole
from .config import LossWeights, SolverConfig
from .architectures import CORE
from .ray import (
    EffectivePinholeCamera,
    effective_camera_bearing_field,
    fit_effective_pinhole_camera,
    ray_cosine_loss,
)
from .rotations import geodesic_distance, matrix_mse
from .precision import fp32_geometry
from .fusion import FusionConfig
from .motion import motion_errors


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    active = torch.broadcast_to(mask.bool(), value.shape)
    # Multiplication is not a valid mask for NaN/Inf (`0 * NaN == NaN`).
    # Reduction only: callers must mask undefined targets BEFORE nonlinear ops
    # too, since masking a NaN loss here does not repair its backward graph.
    safe = torch.where(active, value, torch.zeros((), device=value.device, dtype=value.dtype))
    return safe.sum() / active.sum().clamp_min(1).to(value.dtype)


def masked_clip_mean(value: Tensor, mask: Tensor) -> Tensor:
    """Mean of per-clip valid-item means; empty clips contribute zero.

    This preserves the batch=1 + accumulation objective when clips are
    repacked into equal-size microbatches or equal-size DDP rank batches.
    """

    if value.ndim < 2:
        raise ValueError("masked_clip_mean requires [B,...] values")
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    active = torch.broadcast_to(mask.bool(), value.shape)
    safe = torch.where(active, value, torch.zeros((), device=value.device, dtype=value.dtype))
    numerator = safe.flatten(1).sum(1)
    denominator = active.flatten(1).sum(1).clamp_min(1).to(value.dtype)
    return (numerator / denominator).mean()


@fp32_geometry
def camera_fit_supervision(
    camera: EffectivePinholeCamera,
    target_ray_field: Tensor,
    valid_ray: Tensor | None,
    config: SolverConfig,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Separate numerical validity, approximation quality and supervision.

    Targets and their fitted camera are detached. The optional effective-camera
    target is an independent ablation, not an assertion that a fisheye is a
    pinhole camera. Loss targets do not alter the selected architecture's
    inference gate: Fusion checks RMS compatibility, whereas Core does not.
    """

    with torch.no_grad():
        rays = target_ray_field.detach().float()
        if valid_ray is not None:
            active = valid_ray.bool()
            while active.ndim < rays.ndim:
                active = active.unsqueeze(-1)
            rays = torch.where(active, rays, rays.new_tensor([0., 0., 1.]))
        if not torch.isfinite(rays).all():
            raise ValueError("non-finite valid camera-fit ray target")
        rays = F.normalize(rays, dim=-1, eps=1e-6)
        target_camera = fit_effective_pinhole_camera(rays, config)
        if config.camera_fit_target == "core_bearings" and config.architecture == CORE:
            eligible = torch.ones_like(target_camera.valid)
            bearings = rays[..., :2] / rays[..., 2:].abs().clamp_min(1e-8)
        elif config.camera_fit_target == "pinhole_compatible":
            eligible = target_camera.valid
            bearings = rays[..., :2] / rays[..., 2:].abs().clamp_min(1e-8)
        elif config.camera_fit_target == "effective_camera":
            eligible = target_camera.numerical_valid
            bearings = effective_camera_bearing_field(target_camera, rays.shape[1], rays.shape[2])
        elif config.camera_fit_target == "raw_bearings":
            eligible = target_camera.numerical_valid
            bearings = rays[..., :2] / rays[..., 2:].abs().clamp_min(1e-8)
        else:
            raise ValueError("unknown camera_fit_target")
    numerical = camera.numerical_valid if camera.numerical_valid is not None else camera.valid
    mask = (numerical & eligible)[:, None, None].expand(rays.shape[:-1])
    if valid_ray is not None:
        valid_ray = valid_ray.bool()
        while valid_ray.ndim < mask.ndim:
            valid_ray = valid_ray.unsqueeze(-1)
        mask = mask & valid_ray
    activity = {
        "predicted_numerical_valid_fraction": numerical.float().mean(),
        "target_compatible_fraction": (
            target_camera.numerical_valid
            & (target_camera.rms_normalized <= config.camera_fit_max_rms_normalized)
        ).float().mean(),
        "supervised_ray_fraction": mask.float().mean(),
    }
    return bearings, mask, activity


@fp32_geometry
def camera_fit_bearing_loss(
    camera: EffectivePinholeCamera,
    target_ray_field: Tensor,
    valid_ray: Tensor | None,
    loss_norm: str,
    config: SolverConfig = SolverConfig(),
) -> Tensor:
    """Camera-fit loss with gradient routed only through fitted parameters."""

    fitted_bearings = effective_camera_bearing_field(
        camera,
        target_ray_field.shape[1],
        target_ray_field.shape[2],
    )
    target_bearings, mask, _ = camera_fit_supervision(camera, target_ray_field, valid_ray, config)
    difference = fitted_bearings - target_bearings
    if loss_norm == "l1":
        per_pixel = difference.abs().sum(-1)
    elif loss_norm == "smooth_l1":
        per_pixel = F.smooth_l1_loss(
            fitted_bearings,
            target_bearings,
            reduction="none",
        ).sum(-1)
    else:
        raise ValueError("camera_fit_loss_norm must be l1 or smooth_l1")
    return masked_clip_mean(per_pixel, mask)


def camera_fit_warmup_factor(optimizer_step: int | None, warmup_steps: int) -> float:
    if optimizer_step is None or warmup_steps <= 0:
        return 1.0
    return min(max(float(optimizer_step) / float(warmup_steps), 0.0), 1.0)


@dataclass
class HandPrismPrediction:
    global_rotation: Tensor
    articulation: Tensor
    betas: Tensor
    joints_root_direct: Tensor
    joints_root_mano: Tensor
    joints_camera: Tensor
    translation: Tensor
    anchors_2d: Tensor
    existence_logits: Tensor
    visibility_logits: Tensor
    ray_field: Tensor
    camera_fit: EffectivePinholeCamera | None = None
    wrist_prior: Tensor | None = None
    wrist_log_scale: Tensor | None = None
    reliability_logits: Tensor | None = None
    log_depth: Tensor | None = None


@dataclass
class HandPrismTarget:
    global_rotation: Tensor
    articulation: Tensor
    betas: Tensor
    joints_root: Tensor
    joints_camera: Tensor
    translation: Tensor
    joints_2d: Tensor
    existence: Tensor
    visibility: Tensor
    ray_field: Tensor
    valid_hand: Tensor
    valid_mano: Tensor
    valid_joints_3d: Tensor
    valid_joints_2d: Tensor
    valid_ray: Tensor | None = None
    timestamps: Tensor | None = None
    in_frame: Tensor | None = None
    visibility_valid: Tensor | None = None
    synthetic_occluded: Tensor | None = None


def safe_annotations(target: HandPrismTarget, *, masked_visibility: bool) -> HandPrismTarget:
    """Replace only unsupervised annotations before building the loss graph.

    Valid nonfinite labels are errors, never silently repaired. Source tensors
    are not mutated. Rotations use identity, rays use the optical axis, and
    other missing labels use zero. Timestamps keep their explicit gap masks.
    """
    hand = target.valid_hand.bool()
    mano = hand & target.valid_mano.bool()
    joint = hand[..., None] & target.valid_joints_3d.bool()
    identity = torch.eye(3, dtype=target.global_rotation.dtype, device=hand.device)
    entries = {
        "global_rotation": (mano, identity),
        "articulation": (mano[..., None], identity),
        "betas": (mano, 0.),
        "joints_root": (joint, 0.),
        "joints_camera": (joint & mano[..., None], 0.),
        "translation": (mano, 0.),
        "joints_2d": (hand[..., None] & target.valid_joints_2d.bool(), 0.),
        "existence": (torch.ones_like(hand), 0.),
        "visibility": (target.visibility_valid if masked_visibility and target.visibility_valid is not None
                       else torch.ones_like(hand), 0.),
        "ray_field": (target.valid_ray if target.valid_ray is not None
                      else torch.ones(target.ray_field.shape[:-1], device=hand.device, dtype=torch.bool),
                      target.ray_field.new_tensor([0., 0., 1.])),
    }
    cleaned, checks = {}, []
    for name, (mask, fill) in entries.items():
        value = getattr(target, name)
        mask = mask.bool()
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        cleaned[name] = torch.where(mask, value, torch.as_tensor(fill, device=value.device, dtype=value.dtype))
        checks.append(torch.isfinite(cleaned[name]).all())
    # One host synchronization for all target fields, not one per annotation.
    if not torch.stack(checks).all():
        invalid = [name for name, value in cleaned.items() if not torch.isfinite(value).all()]
        raise ValueError(f"non-finite valid ground truth: {invalid}")
    return replace(target, **cleaned)


class HandPrismLoss(nn.Module):
    def __init__(self, weights: LossWeights = LossWeights(),
                 fusion_config: FusionConfig = FusionConfig()) -> None:
        super().__init__()
        self.weights = weights
        self.fusion_config = fusion_config

    @fp32_geometry
    def forward(
        self,
        prediction: HandPrismPrediction,
        target: HandPrismTarget,
        intrinsics: Tensor,
        image_size: Tensor,
        distortion: Tensor | None = None,
        camera_model: str = PINHOLE,
        camera_parameters: Tensor | None = None,
        source_image_size: Tensor | None = None,
        solver: str = "standard",
        optimizer_step: int | None = None,
        camera_fit_warmup_steps: int = 500,
        camera_fit_loss_norm: str = "l1",
        camera_fit_config: SolverConfig = SolverConfig(),
    ) -> dict[str, Tensor]:
        outputs = [getattr(prediction, field.name) for field in fields(prediction)]
        if not torch.stack([torch.isfinite(value).all() for value in outputs if isinstance(value, Tensor)]).all():
            raise RuntimeError("non-finite model prediction; annotation masks cannot hide model errors")
        # Quality supervision also uses finite OOS projections, independently
        # of the in-frame 2D loss mask. Keep that GT branch detached and safe.
        quality_uv = target.joints_2d.detach()
        quality_valid = torch.isfinite(quality_uv).all(-1)
        quality_uv = torch.where(quality_valid[..., None], quality_uv, 0.)
        target = safe_annotations(target, masked_visibility=self.fusion_config.enabled)
        valid_hand = target.valid_hand.bool()
        valid_mano = valid_hand & target.valid_mano.bool()
        valid_3d = target.valid_joints_3d.bool() & valid_hand.unsqueeze(-1)
        valid_2d = target.valid_joints_2d.bool() & valid_hand.unsqueeze(-1)
        valid_mano_3d = valid_3d & valid_mano.unsqueeze(-1)

        rotation_geo = masked_clip_mean(
            geodesic_distance(prediction.global_rotation, target.global_rotation), valid_mano
        ) + masked_clip_mean(
            geodesic_distance(prediction.articulation, target.articulation),
            valid_mano.unsqueeze(-1),
        )
        rotation_mse = masked_clip_mean(
            matrix_mse(prediction.global_rotation, target.global_rotation), valid_mano
        ) + masked_clip_mean(
            matrix_mse(prediction.articulation, target.articulation), valid_mano.unsqueeze(-1)
        )
        shape = masked_clip_mean(
            (prediction.betas[:, None] - target.betas).abs().mean(-1), valid_mano
        )
        joints_root = masked_clip_mean(
            (prediction.joints_root_direct - target.joints_root).abs().sum(-1), valid_3d
        )
        joints_camera = masked_clip_mean(
            (prediction.joints_camera - target.joints_camera).abs().sum(-1),
            valid_mano_3d,
        )
        wrist = masked_clip_mean(
            (prediction.joints_camera[..., 0, :] - target.joints_camera[..., 0, :]).abs().sum(-1),
            valid_mano & target.valid_joints_3d[..., 0].bool(),
        )
        anchors_2d = masked_clip_mean(
            (prediction.anchors_2d - target.joints_2d).abs().sum(-1), valid_2d
        )
        projected = project_camera(
            prediction.joints_camera,
            intrinsics,
            image_size,
            distortion,
            camera_model=camera_model,
            camera_parameters=camera_parameters,
            source_image_size=source_image_size,
        )
        predicted_front = prediction.joints_camera[..., 2] > 0.01
        reprojection_joints = masked_clip_mean(
            (projected - target.joints_2d).abs().sum(-1),
            valid_2d & target.valid_mano.unsqueeze(-1) & predicted_front,
        )
        reprojection_wrist = masked_clip_mean(
            (projected[..., 0, :] - target.joints_2d[..., 0, :]).abs().sum(-1),
            valid_hand
            & target.valid_mano.bool()
            & target.valid_joints_2d[..., 0].bool()
            & predicted_front[..., 0],
        )
        translation = masked_clip_mean(
            (prediction.translation - target.translation).abs().sum(-1), valid_mano
        )
        existence = F.binary_cross_entropy_with_logits(
            prediction.existence_logits,
            target.existence.to(prediction.existence_logits.dtype),
            reduction="none",
        ).mean()
        visibility_values = F.binary_cross_entropy_with_logits(
            prediction.visibility_logits,
            target.visibility.to(prediction.visibility_logits.dtype),
            reduction="none",
        )
        visibility = (masked_clip_mean(visibility_values, target.visibility_valid)
                      if self.fusion_config.enabled and target.visibility_valid is not None
                      else visibility_values.mean())

        if prediction.joints_camera.shape[1] >= 3:
            acceleration_value = (
                (
                    prediction.joints_camera[:, 2:]
                    - 2.0 * prediction.joints_camera[:, 1:-1]
                    + prediction.joints_camera[:, :-2]
                )
                .abs()
                .sum(-1)
            )
            acceleration_mask = (
                valid_mano_3d[:, 2:] & valid_mano_3d[:, 1:-1] & valid_mano_3d[:, :-2]
            )
            acceleration = masked_clip_mean(acceleration_value, acceleration_mask)
        else:
            acceleration = prediction.joints_camera.sum() * 0.0
        if target.valid_ray is None:
            ray = ray_cosine_loss(prediction.ray_field, target.ray_field)
        else:
            per_ray = 1.0 - (
                F.normalize(prediction.ray_field, dim=-1, eps=1e-6)
                * F.normalize(target.ray_field, dim=-1, eps=1e-6)
            ).sum(-1)
            ray = masked_clip_mean(per_ray, target.valid_ray)

        camera_fit = prediction.ray_field.sum() * 0.0
        camera_fit_factor = 0.0
        if solver == "kfree":
            if prediction.camera_fit is None:
                raise ValueError("K-free camera-fit loss requires a fitted effective camera")
            camera_fit = camera_fit_bearing_loss(
                prediction.camera_fit,
                target.ray_field,
                target.valid_ray,
                camera_fit_loss_norm,
                camera_fit_config,
            )
            camera_fit_factor = camera_fit_warmup_factor(
                optimizer_step,
                camera_fit_warmup_steps,
            )
        elif solver != "standard":
            raise ValueError("solver must be standard or kfree")

        terms = {
            "rotation_geodesic": rotation_geo,
            "rotation_matrix": rotation_mse,
            "shape": shape,
            "joints_root": joints_root,
            "joints_camera": joints_camera,
            "wrist": wrist,
            "anchors_2d": anchors_2d,
            "reprojection_joints": reprojection_joints,
            "reprojection_wrist": reprojection_wrist,
            "translation": translation,
            "existence": existence,
            "visibility": visibility,
            "acceleration": acceleration,
            "ray": ray,
            "camera_fit": camera_fit,
        }
        zero = prediction.joints_root_direct.sum() * 0.
        extra = {key: zero for key in (
            "joints_root_mano", "direct_mano_consistency", "wrist_prior", "log_depth", "reliability",
            "velocity_error", "acceleration_error", "wrist_velocity_error", "wrist_acceleration_error")}
        if self.weights.joints_root_mano:
            extra["joints_root_mano"] = masked_clip_mean(
                (prediction.joints_root_mano - target.joints_root).abs().sum(-1), valid_mano_3d)
        if self.weights.direct_mano_consistency:
            # Start only after the warmup; detach neither branch. Both retain
            # their own GT losses, so consistency cannot replace supervision.
            warmup = self.fusion_config.consistency_warmup_steps
            factor = camera_fit_warmup_factor(
                None if optimizer_step is None else max(0, optimizer_step - warmup), max(warmup, 1))
            extra["direct_mano_consistency"] = factor * masked_clip_mean(
                (prediction.joints_root_direct - prediction.joints_root_mano).abs().sum(-1), valid_mano_3d)
        if self.weights.wrist_prior:
            if prediction.wrist_prior is None or prediction.wrist_log_scale is None:
                raise ValueError("wrist_prior loss requires the independent wrist head")
            valid_wrist = valid_mano_3d[..., 0]
            truth = torch.where(valid_wrist[..., None], target.joints_camera[..., 0, :], 0.)
            scale = prediction.wrist_log_scale
            nll = ((prediction.wrist_prior - truth).abs() * (-scale).exp() + scale).sum(-1)
            extra["wrist_prior"] = masked_clip_mean(nll, valid_wrist)
        if self.weights.log_depth:
            if prediction.log_depth is None:
                raise ValueError("log_depth loss requires the decoder depth head")
            # The depth head predicts camera-space wrist Z, not Euclidean
            # range. This supervision remains active when PnP uses the prior.
            depth = target.joints_camera[..., 0, 2]
            valid_depth = valid_mano_3d[..., 0] & (depth > .01)
            truth = torch.where(valid_depth, depth, 1.).log()
            extra["log_depth"] = masked_clip_mean(
                (prediction.log_depth.squeeze(-1) - truth).abs(), valid_depth)
        if self.weights.reliability:
            if prediction.reliability_logits is None or target.in_frame is None:
                raise ValueError("reliability loss requires joint logits and in_frame labels")
            with torch.no_grad():
                pixel_error = ((prediction.anchors_2d - quality_uv)
                               * image_size[:, None, None, [1, 0]]).norm(dim=-1)
                quality = torch.exp(-pixel_error / 8.)
                quality = quality * target.in_frame
                if target.synthetic_occluded is not None:
                    quality = quality * ~target.synthetic_occluded.bool()
                # Geometric accuracy is the calibration target; do not claim
                # that projected in-frame points are observed/unoccluded GT.
                mask = valid_3d & quality_valid
            extra["reliability"] = masked_clip_mean(F.binary_cross_entropy_with_logits(
                prediction.reliability_logits, quality, reduction="none"), mask)
        motion_keys = ("velocity_error", "acceleration_error", "wrist_velocity_error", "wrist_acceleration_error")
        if any(getattr(self.weights, name) for name in motion_keys):
            if target.timestamps is None:
                raise ValueError("timestamp-aware motion losses cannot use inferred unit frame spacing")
            for pred, truth, mask, keys in (
                (prediction.joints_root_mano, target.joints_root, valid_mano_3d, motion_keys[:2]),
                (prediction.joints_camera[..., 0, :], target.joints_camera[..., 0, :], valid_mano_3d[..., 0], motion_keys[2:]),
            ):
                vel, vm, acc, am = motion_errors(pred, truth, mask, target.timestamps,
                                                self.fusion_config.max_motion_gap_s)
                extra[keys[0]], extra[keys[1]] = masked_clip_mean(vel, vm), masked_clip_mean(acc, am)
        terms.update(extra)
        total = sum(
            getattr(self.weights, name) * value
            for name, value in terms.items()
            if name != "camera_fit"
        )
        total = total + self.weights.camera_fit * camera_fit_factor * camera_fit
        return {"total": total, **terms}
