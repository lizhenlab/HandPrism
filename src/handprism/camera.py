"""Differentiable camera projections used by training and evaluation."""

from __future__ import annotations

import torch
from torch import Tensor


PINHOLE = "pinhole"
FISHEYE624_UPRIGHT = "fisheye624_upright"


def _batch_view(value: Tensor, target_ndim: int) -> Tensor:
    return value.view((value.shape[0],) + (1,) * (target_ndim - 2))


def project_pinhole(
    points: Tensor,
    intrinsics: Tensor,
    image_size: Tensor,
    distortion: Tensor | None = None,
) -> Tensor:
    """Project to normalized image coordinates with optional OpenCV dist8."""

    from .ray import distort_bearings

    points = points.float()
    intrinsics = intrinsics.float()
    image_size = image_size.float()
    distortion = distortion.float() if distortion is not None else None
    batch = points.shape[0]
    view = (batch,) + (1,) * (points.ndim - 2)
    z = points[..., 2].clamp_min(1e-8)
    bearings = (points[..., :2] / z.unsqueeze(-1)).clamp(-10.0, 10.0)
    if distortion is not None:
        bearings = distort_bearings(bearings, distortion)
    u = intrinsics[:, 0, 0].view(view) * bearings[..., 0] + intrinsics[:, 0, 2].view(view)
    v = intrinsics[:, 1, 1].view(view) * bearings[..., 1] + intrinsics[:, 1, 2].view(view)
    return torch.stack(
        (u / image_size[:, 1].view(view), v / image_size[:, 0].view(view)), dim=-1
    )


def project_fisheye624_upright(
    points: Tensor,
    parameters: Tensor,
    source_image_size: Tensor,
    image_size: Tensor,
) -> Tensor:
    """Project upright camera points through Project Aria Fisheye624.

    ``parameters`` follows Project Aria's 15-value order:
    ``f,cx,cy,k0..k5,p0,p1,s0..s3``.  HOT3D RGB is stored in the sensor
    orientation; its training image is rotated clockwise and resized.  This
    function includes that image transform while remaining differentiable.
    """

    points = points.float()
    parameters = parameters.float()
    source_image_size = source_image_size.float()
    image_size = image_size.float()
    if parameters.ndim != 2 or parameters.shape[-1] != 15:
        raise ValueError("Fisheye624 parameters must be [B,15]")
    if source_image_size.shape != (points.shape[0], 2):
        raise ValueError("source_image_size must be [B,2] as (H,W)")
    if image_size.shape != (points.shape[0], 2):
        raise ValueError("image_size must be [B,2] as (H,W)")

    # The loader represents the clockwise-upright camera as
    # R=[[0,-1,0],[1,0,0],[0,0,1]] times the native camera.  Projection needs
    # the inverse transform.
    rotation_upright_from_source = points.new_tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    source = torch.einsum(
        "ij,b...j->b...i", rotation_upright_from_source.transpose(0, 1), points
    )
    x, y, z = source.unbind(-1)
    # A tiny value inside the square root avoids the undefined derivative of
    # hypot at the optical axis while preserving the analytic limit.
    radius_xy = torch.sqrt(x.square() + y.square() + 1e-16)
    theta = torch.atan2(radius_xy, z.clamp_min(1e-4))
    theta2 = theta.square()
    radial = torch.ones_like(theta)
    power = theta2
    parameter_view = (parameters.shape[0],) + (1,) * (points.ndim - 2)
    for index in range(6):
        radial = radial + parameters[:, 3 + index].view(parameter_view) * power
        power = power * theta2
    radius = theta * radial
    radial_u = radius * x / radius_xy
    radial_v = radius * y / radius_xy
    radius2 = radial_u.square() + radial_v.square()
    radius4 = radius2.square()
    p0 = parameters[:, 9].view(parameter_view)
    p1 = parameters[:, 10].view(parameter_view)
    tangent_u = p0 * (2.0 * radial_u.square() + radius2) + 2.0 * p1 * radial_u * radial_v
    tangent_v = p1 * (2.0 * radial_v.square() + radius2) + 2.0 * p0 * radial_u * radial_v
    prism_u = parameters[:, 11].view(parameter_view) * radius2 + parameters[:, 12].view(
        parameter_view
    ) * radius4
    prism_v = parameters[:, 13].view(parameter_view) * radius2 + parameters[:, 14].view(
        parameter_view
    ) * radius4
    focal = parameters[:, 0].view(parameter_view)
    source_u = focal * (radial_u + tangent_u + prism_u) + parameters[:, 1].view(
        parameter_view
    )
    source_v = focal * (radial_v + tangent_v + prism_v) + parameters[:, 2].view(
        parameter_view
    )

    source_height = _batch_view(source_image_size[:, 0], points.ndim)
    source_width = _batch_view(source_image_size[:, 1], points.ndim)
    target_height = _batch_view(image_size[:, 0], points.ndim)
    target_width = _batch_view(image_size[:, 1], points.ndim)
    # Pixel-center-consistent clockwise rotation followed by bilinear resize.
    upright_u = (source_height - 0.5 - source_v) * target_width / source_height - 0.5
    upright_v = (source_u + 0.5) * target_height / source_width - 0.5
    return torch.stack((upright_u / target_width, upright_v / target_height), dim=-1)


def project_camera(
    points: Tensor,
    intrinsics: Tensor,
    image_size: Tensor,
    distortion: Tensor | None = None,
    *,
    camera_model: str = PINHOLE,
    camera_parameters: Tensor | None = None,
    source_image_size: Tensor | None = None,
) -> Tensor:
    if camera_model == PINHOLE:
        return project_pinhole(points, intrinsics, image_size, distortion)
    if camera_model == FISHEYE624_UPRIGHT:
        if camera_parameters is None or source_image_size is None:
            raise ValueError("Fisheye624 projection requires parameters and source image size")
        return project_fisheye624_upright(
            points, camera_parameters, source_image_size, image_size
        )
    raise ValueError(f"unsupported camera model {camera_model!r}")
