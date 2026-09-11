"""Optional Fusion refinements; no dependencies on dataset names or GT cameras.

The all-disabled configuration preserves the previous Fusion computation.
Each experimental change is separately represented in the run configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .rotations import rotation_6d_to_matrix
from .precision import fp32_geometry

if TYPE_CHECKING:
    from .decoder import HandPrismDecoderOutput


@dataclass(frozen=True)
class FusionConfig:
    final_readout: bool = False
    local_rgb: bool = False
    joint_mano: bool = False
    temporal_wrist: bool = False
    reliability: bool = False
    edge_quality: bool = False
    local_resolution: int = 128
    local_chunk_size: int = 8
    local_source_long_side: int = 1408
    roi_min_extent: float = 0.12
    roi_expansion: float = 1.8
    roi_teacher_steps: int = 1000
    consistency_warmup_steps: int = 1000
    max_motion_gap_s: float = 0.15

    @classmethod
    def from_dict(cls, value: dict | None) -> "FusionConfig":
        value = value or {}
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown Fusion settings: {sorted(unknown)}")
        result = cls(**value)
        for key in ("final_readout", "local_rgb", "joint_mano", "temporal_wrist",
                    "reliability", "edge_quality"):
            if type(getattr(result, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("local_resolution", "local_chunk_size", "local_source_long_side"):
            if type(getattr(result, key)) is not int or getattr(result, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if result.local_resolution < 16:
            raise ValueError("local_resolution must be at least 16")
        for key in ("roi_teacher_steps", "consistency_warmup_steps"):
            if type(getattr(result, key)) is not int or getattr(result, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not (0 < result.roi_min_extent <= 1 and 1 <= result.roi_expansion <= 4):
            raise ValueError("invalid normalized ROI size/expansion")
        if not math.isfinite(result.max_motion_gap_s) or result.max_motion_gap_s <= 0:
            raise ValueError("max_motion_gap_s must be finite and positive")
        return result

    @property
    def enabled(self) -> bool:
        return any((self.final_readout, self.local_rgb, self.joint_mano,
                    self.temporal_wrist, self.reliability, self.edge_quality))


def zero_last(module: nn.Sequential) -> nn.Sequential:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


def mlp(width: int, output: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, output))


class LocalRGB(nn.Module):
    """Read per-frame RGB crops, not enlarged low-resolution DiT features.

    Normalized ROI coordinates refer to the same full camera image as anchors.
    Cropping never changes global rays/calibration. Invalid ROIs return zero
    residuals; GT ROIs are only accepted while explicitly training.
    """

    def __init__(self, width: int, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.GroupNorm(4, 24), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(64 * 16, width), nn.GELU(),
        )
        # Pose (global + 15 articulation), 21 x UV, 21 x XYZ root residuals.
        self.residual = zero_last(mlp(width * 2, 96 + 42 + 63))

    @fp32_geometry
    def forward(self, rgb: Tensor, anchors: Tensor, hand: Tensor,
                teacher: Tensor | None = None, teacher_valid: Tensor | None = None,
                teacher_fraction: float = 0.0,
                image_size: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if rgb.ndim != 5 or rgb.shape[:3] != (hand.shape[0], 3, hand.shape[1]):
            raise ValueError("local RGB must be [B,3,T,H,W] aligned with the output frames")
        if image_size is None or image_size.shape != (hand.shape[0], 2):
            raise ValueError("ROI mapping requires global image_size [B,2]")
        uv = anchors.detach().float()
        good = torch.isfinite(uv).all((-1, -2))
        lower, upper = uv.amin(-2), uv.amax(-2)
        center = (lower + upper) * 0.5
        extent = ((upper - lower) * self.config.roi_expansion).clamp(
            min=self.config.roi_min_extent, max=1.5)
        if teacher is not None:
            if not self.training:
                raise ValueError("GT ROIs are forbidden during validation/inference")
            if teacher_valid is None or teacher.shape != anchors.shape:
                raise ValueError("teacher ROI requires joint-aligned valid masks")
            valid = teacher_valid.bool() & torch.isfinite(teacher).all(-1)
            lo = torch.where(valid[..., None], teacher, torch.inf).amin(-2)
            hi = torch.where(valid[..., None], teacher, -torch.inf).amax(-2)
            use = valid.sum(-1) >= 3
            tc = torch.nan_to_num((lo + hi) * 0.5)
            te = torch.nan_to_num((hi - lo) * self.config.roi_expansion).clamp(
                self.config.roi_min_extent, 1.5)
            # One choice and one normalized jitter per clip/hand, shared over
            # time. Interpolating distant box centers would crop background.
            # The RNG is checkpointed; GT is never used at evaluation.
            fraction = float(max(0, min(1, teacher_fraction)))
            if fraction > 0:
                shape = (tc.shape[0], 1, tc.shape[2], 1)
                choose = torch.rand(shape, device=tc.device) < fraction
                jitter = (torch.rand((*shape[:-1], 2), device=tc.device) - .5) * .2
                tc = tc + jitter * te
                choose = choose & use[..., None]
                center = torch.where(choose, tc, center)
                extent = torch.where(choose, te, extent)
                good = torch.where(choose.squeeze(-1), use, good)
        good = good & ((center > -extent / 2) & (center < 1 + extent / 2)).all(-1)
        center = torch.nan_to_num(center, nan=0.5).clamp(-2, 3)
        extent = torch.nan_to_num(extent, nan=self.config.roi_min_extent).clamp(
            self.config.roi_min_extent, 1.5)
        axis = (torch.arange(self.config.local_resolution, device=rgb.device).float() + .5)
        axis = axis / self.config.local_resolution - .5
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        unit = torch.stack((xx, yy), -1)
        grid = center[..., None, None, :] + extent[..., None, None, :] * unit
        # Camera UV is pixel-center-index / global size. grid_sample with
        # align_corners=False instead uses image-edge fractions. The half
        # GLOBAL pixel shift maps the same point onto native-detail pixels.
        shift = .5 / image_size[:, [1, 0]].float()
        grid = grid + shift[:, None, None, None, None, :]
        flat_grid = grid.flatten(0, 2) * 2 - 1
        flat_hand = hand.flatten(0, 2)
        # Chunk conversion of native-resolution uint8 frames bounds temporary memory.
        source = rgb.permute(0, 2, 1, 3, 4).flatten(0, 1)
        result = []
        for start in range(0, flat_grid.shape[0], self.config.local_chunk_size):
            stop = min(start + self.config.local_chunk_size, flat_grid.shape[0])
            frame_index = torch.arange(start, stop, device=rgb.device) // 2
            images = source[frame_index].float()
            if rgb.dtype == torch.uint8:
                images = images / 127.5 - 1
            crops = F.grid_sample(images, flat_grid[start:stop], mode="bilinear",
                                  padding_mode="zeros", align_corners=False)
            detail = self.encoder(crops)
            result.append(self.residual(torch.cat((detail, flat_hand[start:stop]), -1)))
        residual = torch.cat(result).reshape(*hand.shape[:-1], -1)
        residual = residual * good[..., None]
        bounds = torch.cat((center - extent / 2, center + extent / 2), -1)
        return residual, good, bounds


class FusionRefinement(nn.Module):
    def __init__(self, width: int, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            self.local = LocalRGB(width, config) if config.local_rgb else None
        with torch.random.fork_rng(devices=[]):
            self.joint_pose = zero_last(mlp(width * 2, 6)) if config.joint_mano else None
        with torch.random.fork_rng(devices=[]):
            self.wrist = mlp(width, 6) if config.temporal_wrist else None
        with torch.random.fork_rng(devices=[]):
            self.quality = nn.Linear(width, 1) if config.reliability else None

    @fp32_geometry
    def forward(self, decoded: "HandPrismDecoderOutput", rgb: Tensor | None,
                teacher: Tensor | None = None, teacher_valid: Tensor | None = None,
                optimizer_step: int | None = None,
                image_size: Tensor | None = None) -> "HandPrismDecoderOutput":
        hand, joint = decoded.hand_features, decoded.joint_features
        if not self.training and teacher is not None:
            raise ValueError("GT ROIs are forbidden during validation/inference")
        pose = torch.cat((decoded.global_rotation_6d[..., None, :], decoded.articulation_6d), -2)
        if self.joint_pose is not None:
            # Native MANO rotation order: wrist, index, middle, pinky, ring,
            # thumb. Evidence joints use OpenPose's thumb-first order.
            indices = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
            evidence = joint[..., indices, :]
            context = hand[..., None, :].expand_as(evidence)
            pose = pose + self.joint_pose(torch.cat((evidence, context), -1)) * 0.1
        if self.local is not None:
            if rgb is None:
                raise ValueError("Fusion local_rgb requires aligned high-resolution RGB input")
            fraction = (max(0., 1 - (optimizer_step or 0) / self.config.roi_teacher_steps)
                        if self.training and self.config.roi_teacher_steps else 0.)
            residual, valid, bounds = self.local(rgb, decoded.anchors_2d, hand,
                                                 teacher if fraction > 0 else None,
                                                 teacher_valid, fraction, image_size)
            pose = pose + residual[..., :96].reshape_as(pose) * .1
            decoded.anchors_2d = decoded.anchors_2d + residual[..., 96:138].reshape_as(
                decoded.anchors_2d).tanh() * .05
            correction = residual[..., 138:].reshape_as(decoded.joints_root_direct) * .01
            decoded.joints_root_direct = decoded.joints_root_direct + correction - correction[..., :1, :]
            decoded.local_roi_valid, decoded.local_roi_bounds = valid, bounds
        decoded.global_rotation_6d = pose[..., 0, :]
        decoded.articulation_6d = pose[..., 1:, :]
        decoded.global_rotation = rotation_6d_to_matrix(decoded.global_rotation_6d)
        decoded.articulation = rotation_6d_to_matrix(decoded.articulation_6d)
        if self.wrist is not None:
            value = self.wrist(hand).float()
            decoded.wrist_prior = torch.cat((value[..., :2], value[..., 2:3].clamp(-6, 4).exp()), -1)
            decoded.wrist_log_scale = value[..., 3:].clamp(-6, 2)
        if self.quality is not None:
            decoded.reliability_logits = self.quality(joint).squeeze(-1).float()
        return decoded
