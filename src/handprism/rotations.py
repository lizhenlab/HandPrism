"""Continuous rotation representation and SO(3) losses."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def rotation_6d_to_matrix(value: Tensor, eps: float = 1e-8) -> Tensor:
    """Gram--Schmidt map from Zhou et al. 6D rotations to SO(3)."""

    if value.shape[-1] != 6:
        raise ValueError("rotation_6d must end in dimension 6")
    first, second = value[..., :3], value[..., 3:]
    basis_1 = F.normalize(first, dim=-1, eps=eps)
    basis_2 = F.normalize(
        second - (basis_1 * second).sum(dim=-1, keepdim=True) * basis_1,
        dim=-1,
        eps=eps,
    )
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-1)


def geodesic_distance(prediction: Tensor, target: Tensor) -> Tensor:
    """SO(3) geodesic distance in radians."""

    if prediction.shape[-2:] != (3, 3) or target.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must end in [3,3]")
    relative = prediction.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    skew_vee = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine_magnitude = 0.5 * skew_vee.norm(dim=-1)
    return torch.atan2(sine_magnitude, cosine)


def matrix_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean(dim=(-1, -2))


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Exponentiate an axis-angle vector without a small-angle branch."""

    if axis_angle.shape[-1] != 3:
        raise ValueError("axis_angle must end in dimension 3")
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    return torch.matrix_exp(skew.reshape(*axis_angle.shape[:-1], 3, 3))
