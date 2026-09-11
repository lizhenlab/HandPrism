"""Typed HandPrism component defaults and geometry constraints.

Production entry points require an explicit architecture and run config;
component defaults alone are not a complete training configuration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackboneConfig:
    feature_dim: int = 3072
    ffn_dim: int = 14336
    layers: int = 30
    tap_block: int = 15
    input_channels: int = 148
    latent_channels: int = 48
    condition_channels: int = 100
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_length: int = 512
    text_dim: int = 4096
    noise_sigma: float = 0.0
    lora_rank: int = 64
    lora_alpha: float = 64.0
    lora_dropout: float = 0.0


@dataclass(frozen=True)
class DecoderConfig:
    feature_dim: int = 3072
    hidden_dim: int = 384
    layers: int = 4
    heads: int = 8
    ffn_dim: int = 1536
    dropout: float = 0.0
    hand_queries: int = 2
    joints_per_hand: int = 21
    register_queries: int = 4
    spatial_pe_height: int = 16
    spatial_pe_width: int = 16
    ray_frequencies: int = 8
    mano_pose_joints: int = 15
    mano_shape_dim: int = 10
    # Opt-in independent ablation, measured in feature cells. Zero is baseline.
    anchor_offset_cells: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.anchor_offset_cells) or not 0 <= self.anchor_offset_cells <= 1:
            raise ValueError("anchor_offset_cells must be finite and in [0,1]")

    @property
    def joint_queries(self) -> int:
        return self.hand_queries * self.joints_per_hand

    @property
    def queries(self) -> int:
        return self.hand_queries + self.joint_queries + self.register_queries


@dataclass(frozen=True)
class SolverConfig:
    architecture: str = "handprism-fusion"
    min_depth_m: float = 0.05
    anchor_margin: float = 0.02
    min_votes: int = 6
    max_rms_px: float = 15.0
    bbox_rms_fraction: float = 0.25
    eps: float = 1e-8
    # Numerical camera-fit bounds in normalized-image coordinates.
    # These values are explicit run settings recorded in the configuration.
    camera_fit_variance_floor: float = 1e-4
    camera_fit_focal_min: float = 0.05
    camera_fit_focal_max: float = 10.0
    # Fusion additionally gates fits by residual; Core reports it only.
    camera_fit_max_rms_normalized: float = 0.01
    camera_fit_target: str = "pinhole_compatible"
    ray_inverse_iterations: int = 8
    ray_inverse_tolerance: float = 1e-4
    robust_iterations: int = 0
    robust_huber_px: float = 8.0
    depth_refine_fraction: float = 0.0

    def __post_init__(self) -> None:
        if type(self.robust_iterations) is not int or not 0 <= self.robust_iterations <= 3:
            raise ValueError("robust_iterations must be an integer in [0,3]")
        if not math.isfinite(self.robust_huber_px) or self.robust_huber_px <= 0:
            raise ValueError("robust_huber_px must be finite and positive")
        if not math.isfinite(self.depth_refine_fraction) or not 0 <= self.depth_refine_fraction <= .25:
            raise ValueError("depth_refine_fraction must be in [0,.25]")


@dataclass(frozen=True)
class LossWeights:
    rotation_geodesic: float = 1.0
    rotation_matrix: float = 1.0
    shape: float = 0.1
    joints_root: float = 10.0
    joints_camera: float = 5.0
    wrist: float = 2.0
    anchors_2d: float = 1.0
    reprojection_joints: float = 1.0
    reprojection_wrist: float = 0.5
    translation: float = 1.0
    existence: float = 0.5
    visibility: float = 0.25
    acceleration: float = 0.5
    ray: float = 1.0
    camera_fit: float = 5.0
    joints_root_mano: float = 0.0
    direct_mano_consistency: float = 0.0
    wrist_prior: float = 0.0
    log_depth: float = 0.0
    reliability: float = 0.0
    velocity_error: float = 0.0
    acceleration_error: float = 0.0
    wrist_velocity_error: float = 0.0
    wrist_acceleration_error: float = 0.0


@dataclass(frozen=True)
class OptimizerConfig:
    steps: int = 20_000
    warmup_steps: int = 200
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    decoder_lr: float = 2e-4
    lora_lr: float = 1e-4
    patch_lr: float = 2e-5
    clips_per_gpu: int = 4


@dataclass(frozen=True)
class HandPrismConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    rgb_frames: int = 81
    latent_frames: int = 21

    def validate(self) -> None:
        if self.backbone.tap_block < 0 or self.backbone.tap_block >= self.backbone.layers:
            raise ValueError("tap_block must use zero-based indexing inside the DiT stack")
        if self.decoder.queries != 48:
            raise ValueError("the configured decoder must contain exactly 48 queries")
        if (
            self.backbone.latent_channels + self.backbone.condition_channels
            != self.backbone.input_channels
        ):
            raise ValueError(
                "latent and condition channels must sum to patch-embedding input channels"
            )
        expected_latent = (self.rgb_frames - 1) // 4 + 1
        if self.latent_frames != expected_latent:
            raise ValueError("Wan causal temporal compression maps T to (T-1)//4+1")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
