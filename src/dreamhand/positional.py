"""Spatial and ray positional encodings."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def normalized_cell_centers(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return `[H,W,2]` cell centers in `(x,y)` normalized coordinates."""

    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


class LearnedSpatialPE(nn.Module):
    def __init__(self, width: int, base_height: int = 16, base_width: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(1, width, base_height, base_width))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, height: int, width: int) -> Tensor:
        value = F.interpolate(
            self.embedding,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return value.flatten(2).transpose(1, 2)


def ray_fourier_features(rays: Tensor, frequencies: int = 8, eps: float = 1e-8) -> Tensor:
    """Encode ray azimuth/elevation with sin/cos at doubling frequencies."""

    rays = F.normalize(rays, dim=-1, eps=eps)
    azimuth = torch.atan2(rays[..., 0], rays[..., 2])
    elevation = torch.atan2(rays[..., 1], rays[..., [0, 2]].norm(dim=-1))
    angles = torch.stack((azimuth, elevation), dim=-1)
    scales = (2.0 ** torch.arange(frequencies, device=rays.device, dtype=rays.dtype)) * math.pi
    phases = angles.unsqueeze(-1) * scales
    return torch.cat((phases.sin(), phases.cos()), dim=-1).flatten(-2)


class RayPE(nn.Module):
    def __init__(self, width: int, frequencies: int = 8) -> None:
        super().__init__()
        input_dim = 2 * 2 * frequencies
        self.frequencies = frequencies
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        # Zero only the output projection. Zeroing both Linear layers creates
        # a dead branch: the first layer emits zero, so neither weight matrix
        # can ever receive a data-dependent gradient. This keeps the initial
        # ray PE exactly zero while allowing the final layer to learn on the
        # first backward pass.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rays: Tensor) -> Tensor:
        return self.net(ray_fourier_features(rays, self.frequencies))
