"""HandPrism multi-task losses with per-clip reduction and capability masks."""

from __future__ import annotations

from dataclasses import dataclass

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


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    active = torch.broadcast_to(mask.bool(), value.shape)
    # Multiplication is not a valid mask for NaN/Inf (`0 * NaN == NaN`).
    # Invalid annotations may be undefined; select the valid branch explicitly.
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
        rays = F.normalize(target_ray_field.detach().float(), dim=-1, eps=1e-6)
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
class DreamHandPrediction:
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


@dataclass
class DreamHandTarget:
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


class DreamHandLoss(nn.Module):
    def __init__(self, weights: LossWeights = LossWeights()) -> None:
        super().__init__()
        self.weights = weights

    @fp32_geometry
    def forward(
        self,
        prediction: DreamHandPrediction,
        target: DreamHandTarget,
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
        visibility = F.binary_cross_entropy_with_logits(
            prediction.visibility_logits,
            target.visibility.to(prediction.visibility_logits.dtype),
            reduction="none",
        ).mean()

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
        total = sum(
            getattr(self.weights, name) * value
            for name, value in terms.items()
            if name != "camera_fit"
        )
        total = total + self.weights.camera_fit * camera_fit_factor * camera_fit
        return {"total": total, **terms}
