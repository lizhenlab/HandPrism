"""Ray prediction, bearing construction, and the mixed-PnP translation solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import SolverConfig
from .architectures import CORE, architecture_spec
from .precision import fp32_geometry


class RayHead(nn.Module):
    """Zero-initialized 1x1 residual ray head: 9,219 parameters at width 3072."""

    def __init__(self, feature_dim: int = 3072) -> None:
        super().__init__()
        self.projection = nn.Conv2d(feature_dim, 3, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    @fp32_geometry
    def forward(self, features: Tensor) -> Tensor:
        """Map `[B,T,H,W,D]` features to one `[B,H,W,3]` clip ray field."""

        if features.ndim != 5:
            raise ValueError("features must have shape [B,T,H,W,D]")
        batch, frames, height, width, channels = features.shape
        field = self.projection(
            features.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
        )
        field = field.reshape(batch, frames, 3, height, width).mean(dim=1)
        # A zero raw vector has no direction and makes atan2-based ray PE
        # gradients undefined. Interpret the zero-initialized head as a
        # residual around the optical axis; trainable parameters are still
        # exactly zero while the initial unit-ray field is finite.
        base = torch.zeros_like(field)
        base[:, 2] = 1.0
        return F.normalize(field + base, dim=1, eps=1e-6).permute(0, 2, 3, 1)


@dataclass
class EffectivePinholeCamera:
    """Clip-level differentiable effective pinhole camera fit.

    ``focal`` and ``principal`` use normalized image coordinates. ``valid``
    is clip-level because the regression uses the complete token grid.
    """

    focal: Tensor
    principal: Tensor
    bearing_variance: Tensor
    valid: Tensor
    numerical_valid: Tensor | None = None
    rms_normalized: Tensor | None = None


def normalized_pixel_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return normalized feature-cell centers as ``[H,W,2]`` (x, y)."""

    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def fit_effective_pinhole_camera(
    ray_field: Tensor,
    config: SolverConfig = SolverConfig(),
) -> EffectivePinholeCamera:
    """Fit ``pixel = focal * bearing + principal`` independently per axis.

    The closed-form regression uses FP32. Core requires finite parameters,
    sufficient bearing variance and a configured focal bracket. Fusion also
    requires positive finite rays and a bounded normalized projection RMS.
    Rejected fits select the direct ray-sampling branch downstream.
    """

    if ray_field.ndim != 4 or ray_field.shape[-1] != 3:
        raise ValueError("ray_field must be [B,H,W,3]")
    if not 0.0 < config.camera_fit_focal_min < config.camera_fit_focal_max:
        raise ValueError("camera-fit focal bracket must be finite, positive, and ordered")
    if config.camera_fit_variance_floor <= 0.0:
        raise ValueError("camera-fit variance floor must be positive")
    if (
        not math.isfinite(config.camera_fit_max_rms_normalized)
        or config.camera_fit_max_rms_normalized <= 0
    ):
        raise ValueError("camera-fit pixel residual limit must be finite and positive")

    architecture_spec(config.architecture)
    rays = F.normalize(ray_field.float(), dim=-1, eps=1e-6)
    height, width = rays.shape[1:3]
    z = rays[..., 2].abs().clamp_min(config.eps)
    bearings = rays[..., :2] / z.unsqueeze(-1)
    pixels = normalized_pixel_grid(
        height,
        width,
        device=rays.device,
        dtype=rays.dtype,
    )
    bearing_mean = bearings.mean(dim=(1, 2))
    pixel_mean = pixels.mean(dim=(0, 1))
    centered_bearings = bearings - bearing_mean[:, None, None]
    centered_pixels = pixels - pixel_mean
    variance = centered_bearings.square().mean(dim=(1, 2))
    covariance = (centered_bearings * centered_pixels.unsqueeze(0)).mean(dim=(1, 2))
    focal = covariance / variance.clamp_min(config.camera_fit_variance_floor)
    principal = pixel_mean.unsqueeze(0) - focal * bearing_mean

    finite = torch.isfinite(focal).all(-1) & torch.isfinite(principal).all(-1)
    numerical_valid = (
        (variance >= config.camera_fit_variance_floor).all(-1)
        & (focal >= config.camera_fit_focal_min).all(-1)
        & (focal <= config.camera_fit_focal_max).all(-1)
        & finite
    )
    if config.architecture != CORE:
        numerical_valid = numerical_valid & torch.isfinite(rays).all(dim=(1, 2, 3))
        numerical_valid = numerical_valid & (rays[..., 2] > config.eps).all(dim=(1, 2))
    residual = bearings * focal[:, None, None] + principal[:, None, None] - pixels
    rms = residual.square().sum(-1).mean(dim=(1, 2)).sqrt()
    valid = numerical_valid if config.architecture == CORE else (
        numerical_valid & (rms <= config.camera_fit_max_rms_normalized)
    )
    # Keep invalid branches finite so torch.where/masked losses cannot leak a
    # NaN into gradients.  ``valid`` preserves the guard decision.
    focal = torch.nan_to_num(focal, nan=1.0, posinf=1.0, neginf=-1.0)
    focal = torch.where(focal.abs() >= config.eps, focal, torch.ones_like(focal))
    principal = torch.nan_to_num(principal, nan=0.5, posinf=1.0, neginf=0.0)
    return EffectivePinholeCamera(
        focal=focal,
        principal=principal,
        bearing_variance=variance,
        valid=valid,
        numerical_valid=numerical_valid,
        rms_normalized=rms,
    )


def bearings_from_effective_camera(
    anchors: Tensor,
    camera: EffectivePinholeCamera,
) -> Tensor:
    """Construct fitted-camera bearings ``(p-c)/f`` at normalized anchors."""

    if anchors.shape[0] != camera.focal.shape[0] or anchors.shape[-1] != 2:
        raise ValueError("anchor and fitted-camera batches must match")
    view = (anchors.shape[0],) + (1,) * (anchors.ndim - 2) + (2,)
    return (anchors.float() - camera.principal.view(view)) / camera.focal.view(view)


def effective_camera_bearing_field(
    camera: EffectivePinholeCamera,
    height: int,
    width: int,
) -> Tensor:
    """Evaluate the fitted camera over its normalized token grid."""

    pixels = normalized_pixel_grid(
        height,
        width,
        device=camera.focal.device,
        dtype=camera.focal.dtype,
    )
    return (pixels.unsqueeze(0) - camera.principal[:, None, None]) / camera.focal[:, None, None]


def kfree_bearings(
    ray_field: Tensor,
    anchors: Tensor,
    config: SolverConfig = SolverConfig(),
) -> tuple[Tensor, EffectivePinholeCamera]:
    """Use fitted bearings when accepted, otherwise sample predicted rays."""

    camera = fit_effective_pinhole_camera(ray_field, config)
    fitted = bearings_from_effective_camera(anchors, camera)
    direct = sample_ray_bearings(ray_field, anchors, config.eps).float()
    valid = camera.valid.view((anchors.shape[0],) + (1,) * (anchors.ndim - 1))
    return torch.where(valid, fitted, direct), camera


# Compatibility alias for existing imports; no architecture is selected here.
ace_kfree_bearings = kfree_bearings


def pinhole_rays(
    intrinsics: Tensor,
    height: int,
    width: int,
    image_size: Tensor,
    distortion: Tensor | None = None,
) -> Tensor:
    """Unit rays through feature-cell centers.

    ``distortion`` follows OpenCV's rational+tangential eight-coefficient
    convention.  When supplied, feature-cell pixels live in the distorted RGB
    image and are iteratively mapped back to physical camera bearings.
    """

    batch = intrinsics.shape[0]
    y = (torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype) + 0.5) / height
    x = (torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pixel_x = xx[None] * image_size[:, 1, None, None]
    pixel_y = yy[None] * image_size[:, 0, None, None]
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]
    bearings = torch.stack(((pixel_x - cx) / fx, (pixel_y - cy) / fy), dim=-1)
    if distortion is not None:
        bearings = undistort_bearings(bearings, distortion)
    rays = torch.cat((bearings, torch.ones_like(bearings[..., :1])), dim=-1)
    return F.normalize(rays, dim=-1)


def distort_bearings(bearings: Tensor, coefficients: Tensor) -> Tensor:
    """Apply OpenCV rational+tangential distortion to normalized bearings."""

    if bearings.shape[-1] != 2 or coefficients.ndim != 2 or coefficients.shape[-1] != 8:
        raise ValueError("bearings must end in 2 and coefficients must be [B,8]")
    if bearings.shape[0] != coefficients.shape[0]:
        raise ValueError("bearing and distortion batches must match")
    view = (coefficients.shape[0],) + (1,) * (bearings.ndim - 2) + (8,)
    k = coefficients.to(device=bearings.device, dtype=bearings.dtype).view(view)
    x, y = bearings.unbind(-1)
    x2, y2, xy = x.square(), y.square(), x * y
    r2 = x2 + y2
    r4, r6 = r2.square(), r2.square() * r2
    radial = (1 + k[..., 0] * r2 + k[..., 1] * r4 + k[..., 4] * r6) / (
        1 + k[..., 5] * r2 + k[..., 6] * r4 + k[..., 7] * r6
    ).clamp_min(1e-8)
    xd = x * radial + 2 * k[..., 2] * xy + k[..., 3] * (r2 + 2 * x2)
    yd = y * radial + 2 * k[..., 3] * xy + k[..., 2] * (r2 + 2 * y2)
    return torch.stack((xd, yd), dim=-1)


def undistort_bearings(
    distorted: Tensor,
    coefficients: Tensor,
    iterations: int = 8,
) -> Tensor:
    """Differentiably invert ``distort_bearings`` by fixed-point refinement."""

    estimate = distorted
    for _ in range(iterations):
        estimate = estimate + distorted - distort_bearings(estimate, coefficients)
    return estimate


def bearings_from_intrinsics(
    anchors: Tensor,
    intrinsics: Tensor,
    image_size: Tensor,
    distortion: Tensor | None = None,
) -> Tensor:
    """Physical `(x/z,y/z)` bearings at normalized image anchors."""

    batch = anchors.shape[0]
    view = (batch,) + (1,) * (anchors.ndim - 2)
    u = anchors[..., 0] * image_size[:, 1].view(view)
    v = anchors[..., 1] * image_size[:, 0].view(view)
    bx = (u - intrinsics[:, 0, 2].view(view)) / intrinsics[:, 0, 0].view(view)
    by = (v - intrinsics[:, 1, 2].view(view)) / intrinsics[:, 1, 1].view(view)
    bearings = torch.stack((bx, by), dim=-1)
    if distortion is not None:
        bearings = undistort_bearings(bearings, distortion)
    return bearings


def sample_ray_bearings(ray_field: Tensor, anchors: Tensor, eps: float = 1e-8) -> Tensor:
    """Bilinearly sample a clip ray field at `[B,T,S,J,2]` anchors."""

    if ray_field.ndim != 4 or ray_field.shape[-1] != 3:
        raise ValueError("ray_field must be [B,H,W,3]")
    batch = ray_field.shape[0]
    leading = anchors.shape[1:-1]
    grid = anchors.reshape(batch, -1, 1, 2) * 2.0 - 1.0
    sampled = F.grid_sample(
        ray_field.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(batch, *leading, 3)
    z = sampled[..., 2].abs().clamp_min(eps)
    return sampled[..., :2] / z.unsqueeze(-1)


def project_with_bearings(points: Tensor, eps: float = 1e-8) -> Tensor:
    return points[..., :2] / points[..., 2:].abs().clamp_min(eps)


@torch.no_grad()
def project_via_ray_field(
    points: Tensor,
    ray_field: Tensor,
    initial_uv: Tensor,
    config: SolverConfig = SolverConfig(),
) -> tuple[Tensor, Tensor]:
    """Invert the *predicted* ray map for the K-free pixel-domain gate.

    Local Newton iterations invert the same bilinear map used for bearings.
    A singular, nonconverged or out-of-grid inverse is rejected, never assigned
    an image-diagonal proxy residual. This discrete acceptance test is detached;
    the closed-form translation and its accepted branch remain differentiable.
    No calibration or camera-type labels are consumed.
    """
    rays = ray_field.detach().float()
    if config.ray_inverse_iterations < 1 or not 0 < config.ray_inverse_tolerance < 1:
        raise ValueError("invalid ray inverse iteration/tolerance settings")
    height, width = rays.shape[1:3]
    uv = initial_uv.detach().float().clone()
    target = project_with_bearings(points.detach().float(), config.eps)
    step = uv.new_tensor([0.25 / width, 0.25 / height])
    lower = uv.new_tensor([0.5 / width, 0.5 / height])
    upper = 1.0 - lower
    valid = (points[..., 2] > config.eps) & torch.isfinite(points).all(-1)
    valid &= (
        torch.isfinite(rays).all(dim=(1, 2, 3)).view((rays.shape[0],) + (1,) * (valid.ndim - 1))
    )
    for _ in range(config.ray_inverse_iterations):
        current = sample_ray_bearings(rays, uv, config.eps)
        dx = step * uv.new_tensor([1.0, 0.0])
        dy = step * uv.new_tensor([0.0, 1.0])
        jx = (sample_ray_bearings(rays, uv + dx) - sample_ray_bearings(rays, uv - dx)) / (
            2 * step[0]
        )
        jy = (sample_ray_bearings(rays, uv + dy) - sample_ray_bearings(rays, uv - dy)) / (
            2 * step[1]
        )
        det = jx[..., 0] * jy[..., 1] - jy[..., 0] * jx[..., 1]
        invertible = torch.isfinite(det) & (det.abs() > 1e-6)
        valid &= invertible
        safe_det = torch.where(invertible, det, torch.ones_like(det))
        error = current - target
        update = torch.stack(
            (
                (jy[..., 1] * error[..., 0] - jy[..., 0] * error[..., 1]) / safe_det,
                (-jx[..., 1] * error[..., 0] + jx[..., 0] * error[..., 1]) / safe_det,
            ),
            dim=-1,
        )
        uv = (uv - torch.nan_to_num(update).clamp(-0.15, 0.15)).clamp(0.0, 1.0)
    error = (sample_ray_bearings(rays, uv) - target).norm(dim=-1)
    valid &= (error <= config.ray_inverse_tolerance) & torch.isfinite(uv).all(-1)
    valid &= ((uv >= lower) & (uv <= upper)).all(-1)
    return torch.nan_to_num(uv), valid


def project_kfree(
    points: Tensor,
    camera: EffectivePinholeCamera,
    ray_field: Tensor,
    initial_uv: Tensor,
    config: SolverConfig = SolverConfig(),
) -> tuple[Tensor, Tensor]:
    """Project with the accepted fit or invert the predicted non-pinhole map."""
    view = (points.shape[0],) + (1,) * (points.ndim - 2) + (2,)
    fitted = project_with_bearings(points) * camera.focal.view(view) + camera.principal.view(view)
    # The gate is nondifferentiable and need not build a projection graph.
    with torch.no_grad():
        output = fitted.detach().clone()
        valid = torch.isfinite(fitted).all(-1) & (points[..., 2] > 0)
        fallback = ~camera.valid
        if fallback.any():
            direct, direct_valid = project_via_ray_field(
                points[fallback], ray_field[fallback], initial_uv[fallback], config
            )
            output[fallback] = direct
            valid[fallback] = direct_valid
        return output, valid


@dataclass
class MixedPnPOutput:
    """Translation and gate diagnostics; interpret RMS using residual_kind.

    The legacy rms_pixels field carries a bearing-diagonal proxy for Core,
    but an actual pixel residual for Fusion. It is not a shared error unit.
    """

    translation: Tensor
    solved: Tensor
    vote_count: Tensor
    rms_pixels: Tensor
    used_fallback: Tensor
    projection_valid: Tensor | None = None
    residual_limit_pixels: Tensor | None = None
    failure_code: Tensor | None = None
    residual_kind: str = "native_pixels"


def mixed_pnp(
    canonical_joints: Tensor,
    anchors: Tensor,
    log_depth: Tensor,
    bearings: Tensor,
    image_size: Tensor,
    config: SolverConfig = SolverConfig(),
    *,
    projector: Callable[[Tensor], tuple[Tensor, Tensor]],
) -> MixedPnPOutput:
    """Depth-conditioned translation solve with voting and wrist fallback.

    Core gates a bearing-diagonal proxy; Fusion gates a calibrated projection
    or predicted-ray inverse in pixel units. Both expose their residual kind.

    Shapes: joints `[B,T,2,21,3]`, anchors/bearings matching joints with final
    dimensions 2, log_depth `[B,T,2,1]`, image_size `[B,2]` as `(H,W)`.
    """

    if canonical_joints.shape[:-1] != anchors.shape[:-1]:
        raise ValueError("canonical_joints and anchors must index identical joints")
    if bearings.shape != anchors.shape:
        raise ValueError("bearings and anchors must have the same shape")
    if log_depth.shape != canonical_joints.shape[:3] + (1,):
        raise ValueError("log_depth must be [B,T,2,1]")

    # The closed-form solve and its threshold decision are sensitive to small
    # sums and ratios. Keep geometry in FP32 even under BF16 autocast; casts
    # remain differentiable back to decoder and ray-head outputs.
    canonical_joints = canonical_joints.float()
    anchors = anchors.float()
    log_depth = log_depth.float()
    bearings = bearings.float()
    image_size = image_size.float()

    depth = log_depth.exp()[..., 0]
    joint_z = canonical_joints[..., 2] + depth.unsqueeze(-1)
    inside = ((anchors >= config.anchor_margin) & (anchors <= 1.0 - config.anchor_margin)).all(-1)
    votes = inside & (joint_z >= config.min_depth_m)
    weight = votes.to(canonical_joints.dtype)

    inv_z = joint_z.clamp_min(config.eps).reciprocal()
    numerator_x = (weight * inv_z * (bearings[..., 0] - canonical_joints[..., 0] * inv_z)).sum(-1)
    numerator_y = (weight * inv_z * (bearings[..., 1] - canonical_joints[..., 1] * inv_z)).sum(-1)
    denominator = (weight * inv_z.square()).sum(-1).clamp_min(config.eps)
    tx = numerator_x / denominator
    ty = numerator_y / denominator
    translation = torch.stack((tx, ty, depth), dim=-1)

    placed = canonical_joints + translation.unsqueeze(-2)
    architecture_spec(config.architecture)
    if config.architecture == CORE:
        # Preserve the trained Core solver's acceptance rule. This is NOT a
        # measured pixel reprojection error; label it explicitly in exports.
        image_diag = image_size.square().sum(-1).sqrt().view(-1, 1, 1, 1)
        residual_pixels = (project_with_bearings(placed) - bearings).norm(dim=-1) * image_diag
        projection_valid = torch.isfinite(residual_pixels).all(-1)
    else:
        with torch.no_grad():
            reprojected, projected_valid = projector(placed.detach())
            if reprojected.shape != anchors.shape or projected_valid.shape != votes.shape:
                raise ValueError("projector must return normalized UV and per-joint validity")
            projected_valid = projected_valid & torch.isfinite(reprojected).all(-1)
            projection_valid = (projected_valid | ~votes).all(-1)
            safe_projected = torch.where(projected_valid.unsqueeze(-1), reprojected, anchors)
            residual_pixels = ((safe_projected - anchors) * image_size[:, None, None, [1, 0]]).norm(
                dim=-1
            )
    vote_count = votes.sum(-1)
    rms = ((residual_pixels.square() * weight).sum(-1) / vote_count.clamp_min(1)).sqrt()

    box_extent = anchors.amax(dim=-2) - anchors.amin(dim=-2)
    box_diag_pixels = (box_extent * image_size[:, None, None, [1, 0]]).norm(dim=-1)
    residual_limit = torch.maximum(
        torch.full_like(box_diag_pixels, config.max_rms_px),
        config.bbox_rms_fraction * box_diag_pixels,
    )
    finite = torch.isfinite(rms) & torch.isfinite(translation).all(-1)
    solved = (vote_count >= config.min_votes) & (rms <= residual_limit) & finite & projection_valid
    # Bitmask: 1 insufficient votes, 2 reprojection too large, 4 inverse failed,
    # 8 nonfinite solve. An unavailable inverse has a finite diagnostic RMS,
    # but projection_valid/failure_code prevent it being interpreted as solved.
    failure_code = (
        (vote_count < config.min_votes).long()
        | ((rms > residual_limit).long() * 2)
        | ((~projection_valid).long() * 4)
        | ((~finite).long() * 8)
    )

    wrist_bearing = bearings[..., 0, :]
    wrist = canonical_joints[..., 0, :]
    fallback_wrist = torch.stack(
        (wrist_bearing[..., 0] * depth, wrist_bearing[..., 1] * depth, depth), dim=-1
    )
    fallback_translation = fallback_wrist - wrist
    translation = torch.where(solved.unsqueeze(-1), translation, fallback_translation)
    return MixedPnPOutput(
        translation=translation,
        solved=solved,
        vote_count=vote_count,
        rms_pixels=rms,
        used_fallback=~solved,
        projection_valid=projection_valid,
        residual_limit_pixels=residual_limit,
        failure_code=failure_code,
        residual_kind="bearing_diagonal_proxy" if config.architecture == CORE else "native_pixels",
    )


def ray_cosine_loss(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = F.normalize(prediction, dim=-1, eps=1e-6)
    target = F.normalize(target, dim=-1, eps=1e-6)
    return (1.0 - (prediction * target).sum(-1)).mean()
