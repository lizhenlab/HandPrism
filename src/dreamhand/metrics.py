"""Hand detection, pose alignment and temporal-jitter evaluation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .camera import PINHOLE, project_camera
from .rotations import geodesic_distance


def on_screen_from_projection(
    joints_camera: Tensor, projected: Tensor, valid: Tensor | None = None
) -> Tensor:
    inside = ((projected >= 0.0) & (projected < 1.0)).all(-1)
    positive = joints_camera[..., 2] > 0.01
    if valid is not None:
        inside = inside & valid.bool()
    return (inside & positive).any(-1)


def on_screen(
    joints_camera: Tensor,
    intrinsics: Tensor,
    image_size: Tensor,
    distortion: Tensor | None = None,
    *,
    camera_model: str = PINHOLE,
    camera_parameters: Tensor | None = None,
    source_image_size: Tensor | None = None,
    valid: Tensor | None = None,
) -> Tensor:
    projected = project_camera(
        joints_camera,
        intrinsics,
        image_size,
        distortion,
        camera_model=camera_model,
        camera_parameters=camera_parameters,
        source_image_size=source_image_size,
    )
    return on_screen_from_projection(joints_camera, projected, valid)


def projected_boxes_from_projection(
    points_camera: Tensor, projected: Tensor, valid: Tensor | None = None
) -> Tensor:
    positive = points_camera[..., 2] > 0.01
    if valid is not None:
        positive = positive & valid.bool()
    minimum = torch.where(
        positive.unsqueeze(-1), projected, torch.full_like(projected, float("inf"))
    ).amin(-2)
    maximum = torch.where(
        positive.unsqueeze(-1), projected, torch.full_like(projected, float("-inf"))
    ).amax(-2)
    return torch.cat((minimum, maximum), dim=-1)


def projected_boxes(
    vertices_camera: Tensor,
    intrinsics: Tensor,
    image_size: Tensor,
    distortion: Tensor | None = None,
    *,
    camera_model: str = PINHOLE,
    camera_parameters: Tensor | None = None,
    source_image_size: Tensor | None = None,
    valid: Tensor | None = None,
) -> Tensor:
    projected = project_camera(
        vertices_camera,
        intrinsics,
        image_size,
        distortion,
        camera_model=camera_model,
        camera_parameters=camera_parameters,
        source_image_size=source_image_size,
    )
    return projected_boxes_from_projection(vertices_camera, projected, valid)


def dilate_boxes(boxes: Tensor, fraction: float = 0.10) -> Tensor:
    center = 0.5 * (boxes[..., :2] + boxes[..., 2:])
    half = 0.5 * (boxes[..., 2:] - boxes[..., :2]) * (1.0 + fraction)
    return torch.cat((center - half, center + half), dim=-1)


def aligned_iou(first: Tensor, second: Tensor) -> Tensor:
    top_left = torch.maximum(first[..., :2], second[..., :2])
    bottom_right = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp_min(0.0).prod(-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp_min(0.0).prod(-1)
    return intersection / (first_area + second_area - intersection).clamp_min(1e-8)


def wrist_aligned_mpjpe(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = prediction - prediction[..., :1, :]
    target = target - target[..., :1, :]
    return (prediction - target).norm(dim=-1).mean(dim=-1)


def procrustes_mpjpe(prediction: Tensor, target: Tensor) -> Tensor:
    """Similarity-aligned MPJPE for arbitrary leading dimensions."""

    pred_mean = prediction.mean(-2, keepdim=True)
    target_mean = target.mean(-2, keepdim=True)
    pred_centered = prediction - pred_mean
    target_centered = target - target_mean
    covariance = pred_centered.transpose(-1, -2) @ target_centered
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones_like(singular)
    correction[..., -1] = torch.det(u @ vh)
    rotation = u @ torch.diag_embed(correction) @ vh
    scale = (singular * correction).sum(-1) / pred_centered.square().sum((-1, -2)).clamp_min(1e-8)
    aligned = scale[..., None, None] * (pred_centered @ rotation) + target_mean
    return (aligned - target).norm(dim=-1).mean(dim=-1)


def masked_wrist_aligned_mpjpe(
    prediction: Tensor, target: Tensor, valid: Tensor
) -> Tensor:
    prediction = prediction - prediction[..., :1, :]
    target = target - target[..., :1, :]
    error = (prediction - target).norm(dim=-1)
    mask = valid.to(error.dtype)
    return (error * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


def masked_procrustes_mpjpe(
    prediction: Tensor, target: Tensor, valid: Tensor
) -> Tensor:
    """Similarity-aligned MPJPE using only valid joints per leading item."""

    leading = prediction.shape[:-2]
    output = prediction.new_zeros(int(torch.tensor(leading).prod()) if leading else 1)
    flat_prediction = prediction.reshape(-1, prediction.shape[-2], 3)
    flat_target = target.reshape_as(flat_prediction)
    flat_valid = valid.reshape(-1, valid.shape[-1]).bool()
    for index, mask in enumerate(flat_valid):
        if int(mask.sum()) < 3:
            output[index] = float("nan")
        else:
            output[index] = procrustes_mpjpe(
                flat_prediction[index, mask], flat_target[index, mask]
            )
    return output.reshape(leading)


def jitter_matched_run_sum_count(joints: Tensor, matched: Tensor) -> tuple[Tensor, int]:
    """Sum/count of joint second differences over consecutive matched frames."""

    total = joints.new_zeros(())
    count = 0
    start: int | None = None
    for index in range(matched.numel() + 1):
        active = index < matched.numel() and bool(matched[index])
        if active and start is None:
            start = index
        if not active and start is not None:
            if index - start >= 3:
                run = joints[start:index]
                values = (run[2:] - 2.0 * run[1:-1] + run[:-2]).norm(dim=-1)
                total = total + values.sum()
                count += values.numel()
            start = None
    return total, count


def jitter_matched_runs(joints: Tensor, matched: Tensor) -> Tensor:
    """Mean second difference over maximal matched runs of length >=3.

    Inputs are one segment/hand: joints `[T,21,3]`, matched `[T]`.
    """

    total, count = jitter_matched_run_sum_count(joints, matched)
    if not count:
        return joints.new_tensor(float("nan"))
    return total / count


@dataclass
class DetectionCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    correct_frames: int = 0
    frames: int = 0

    @property
    def precision(self) -> float:
        return self.true_positive / max(self.true_positive + self.false_positive, 1)

    @property
    def recall(self) -> float:
        return self.true_positive / max(self.true_positive + self.false_negative, 1)

    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / max(self.precision + self.recall, 1e-12)

    @property
    def frame_accuracy(self) -> float:
        return self.correct_frames / max(self.frames, 1)


def global_orientation_error(prediction: Tensor, target: Tensor) -> Tensor:
    return geodesic_distance(prediction, target) * (180.0 / torch.pi)
