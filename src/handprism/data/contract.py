"""Strict cross-dataset tensor contract; converters must satisfy it."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class HandPrismSample:
    dataset: str
    recording_id: str
    frame_indices: Tensor
    video: Tensor
    intrinsics: Tensor
    image_size: Tensor
    global_rotation: Tensor
    articulation: Tensor
    betas: Tensor
    translation: Tensor
    joints_root: Tensor
    joints_camera: Tensor
    joints_2d: Tensor
    existence: Tensor
    visibility: Tensor
    valid_hand: Tensor
    valid_mano: Tensor
    valid_joints_3d: Tensor
    valid_joints_2d: Tensor
    valid_ray: Tensor
    gt_ray_field: Tensor | None = None
    distortion: Tensor | None = None
    camera_model: str = "pinhole"
    camera_parameters: Tensor | None = None
    source_image_size: Tensor | None = None
    timestamps: Tensor | None = None
    timestamp_source: str | None = None
    rgb_high: Tensor | None = None
    in_frame: Tensor | None = None
    observed: Tensor | None = None
    observed_valid: Tensor | None = None
    visibility_valid: Tensor | None = None
    synthetic_occluded: Tensor | None = None


def validate_sample(sample: HandPrismSample) -> None:
    if sample.video.ndim != 4 or sample.video.shape[0] != 3:
        raise ValueError("video must be [3,T,H,W]")
    frames = sample.video.shape[1]
    if sample.timestamps is not None:
        if sample.timestamps.shape != (frames,) or sample.timestamps.dtype != torch.float64:
            raise ValueError("timestamps must be float64 [T] seconds")
        if not torch.isfinite(sample.timestamps).all() or (sample.timestamps.diff() < 0).any():
            raise ValueError("timestamps must be finite and nondecreasing")
        if not sample.timestamp_source:
            raise ValueError("timestamps require provenance; do not silently invent frame rate")
    if sample.rgb_high is not None:
        if (sample.rgb_high.ndim != 4 or sample.rgb_high.shape[:2] != (3, frames)
                or sample.rgb_high.dtype != torch.uint8 or min(sample.rgb_high.shape[-2:]) < 1):
            raise ValueError("rgb_high must be aligned uint8 [3,T,H,W]")
    for key in ("in_frame", "observed", "observed_valid", "synthetic_occluded"):
        value = getattr(sample, key)
        if value is not None and (value.shape != (frames, 2, 21) or value.dtype != torch.bool):
            raise ValueError(f"{key} must be bool [T,2,21]")
    if (sample.observed is None) != (sample.observed_valid is None):
        raise ValueError("observed labels require an explicit observed_valid mask")
    if sample.visibility_valid is not None and (
            sample.visibility_valid.shape != (frames, 2) or sample.visibility_valid.dtype != torch.bool):
        raise ValueError("visibility_valid must be bool [T,2]")
    if sample.frame_indices.shape != (frames,):
        raise ValueError("frame_indices must identify every RGB frame")
    if sample.global_rotation.shape != (frames, 2, 3, 3):
        raise ValueError("global_rotation must be [T,2,3,3]")
    if sample.articulation.shape != (frames, 2, 15, 3, 3):
        raise ValueError("articulation must be [T,2,15,3,3]")
    if sample.betas.shape != (frames, 2, 10):
        raise ValueError("betas must be [T,2,10]; constant shape is checked by the converter")
    if sample.translation.shape != (frames, 2, 3):
        raise ValueError("translation must be [T,2,3] in metres")
    for name in ("joints_root", "joints_camera"):
        if getattr(sample, name).shape != (frames, 2, 21, 3):
            raise ValueError(f"{name} must be [T,2,21,3] in metres")
    if sample.joints_2d.shape != (frames, 2, 21, 2):
        raise ValueError("joints_2d must be normalized [T,2,21,2]")
    if sample.valid_hand.shape != (frames, 2):
        raise ValueError("valid_hand must be [T,2]")
    if sample.valid_mano.shape != (frames, 2):
        raise ValueError("valid_mano must be [T,2]")
    if sample.valid_joints_3d.shape != (frames, 2, 21):
        raise ValueError("valid_joints_3d must be [T,2,21]")
    if sample.valid_joints_2d.shape != (frames, 2, 21):
        raise ValueError("valid_joints_2d must be [T,2,21]")
    if sample.valid_ray.numel() != 1:
        raise ValueError("valid_ray must be a scalar dataset-capability flag")
    if sample.intrinsics.shape not in ((3, 3), (frames, 3, 3)):
        raise ValueError("intrinsics must be clip-constant [3,3] or per-frame [T,3,3]")
    if sample.distortion is not None and sample.distortion.shape not in ((8,), (frames, 8)):
        raise ValueError("distortion must be clip-constant [8] or per-frame [T,8]")
    if sample.gt_ray_field is not None:
        if sample.gt_ray_field.ndim != 3 or sample.gt_ray_field.shape[-1] != 3:
            raise ValueError("gt_ray_field must be [H,W,3]")
    if sample.camera_model not in {"pinhole", "fisheye624_upright"}:
        raise ValueError(f"unsupported camera model {sample.camera_model!r}")
    if sample.camera_model == "fisheye624_upright":
        if sample.camera_parameters is None or sample.camera_parameters.shape != (15,):
            raise ValueError("fisheye624_upright requires 15 camera parameters")
        if sample.source_image_size is None or sample.source_image_size.shape != (2,):
            raise ValueError("fisheye624_upright requires source_image_size [2]")
        if sample.gt_ray_field is None:
            raise ValueError("fisheye624_upright requires a calibrated gt_ray_field")
