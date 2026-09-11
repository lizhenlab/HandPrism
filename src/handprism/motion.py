"""Timestamp-aware motion errors; duplicate frames and large gaps are excluded."""
from __future__ import annotations

import torch
from torch import Tensor


def motion_errors(prediction: Tensor, target: Tensor, valid: Tensor,
                  timestamps: Tensor, max_gap_s: float = .15) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if timestamps.shape != prediction.shape[:2]:
        raise ValueError("timestamps must be [B,T] in seconds")
    # Subtract before downcasting: epoch-scale FP32 timestamps lose frame gaps.
    dt64 = timestamps.double()[:, 1:] - timestamps.double()[:, :-1]
    good = torch.isfinite(dt64) & (dt64 > 1e-6) & (dt64 <= max_gap_s)
    dt = torch.where(good, dt64, torch.ones_like(dt64)).float()
    shape = (*dt.shape, *((1,) * (prediction.ndim - 2)))
    difference = torch.where(valid[..., None], prediction - target, 0.)
    velocity = (difference[:, 1:] - difference[:, :-1]) / dt.reshape(shape)
    vm = valid[:, 1:] & valid[:, :-1] & good.reshape(*dt.shape, *((1,) * (valid.ndim - 2)))
    at = ((dt[:, 1:] + dt[:, :-1]) * .5).reshape(
        dt.shape[0], max(dt.shape[1] - 1, 0), *((1,) * (prediction.ndim - 2)))
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / at
    am = vm[:, 1:] & vm[:, :-1]
    return velocity.norm(dim=-1), vm, acceleration.norm(dim=-1), am
