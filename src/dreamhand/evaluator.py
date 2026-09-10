"""HandPrism metrics with missed-detection penalties and visibility strata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from .camera import project_camera
from .metrics import (
    aligned_iou,
    dilate_boxes,
    global_orientation_error,
    jitter_matched_run_sum_count,
    masked_procrustes_mpjpe,
    masked_wrist_aligned_mpjpe,
    on_screen_from_projection,
    projected_boxes,
)
from .model import DreamHandOutput


STATE_KEYS = (
    "segments",
    "frames",
    "correct_frames",
    "true_positive",
    "false_positive",
    "false_negative",
    "penalty_hands",
    "mpjpe_penalty_sum_mm",
    "pa_penalty_sum_mm",
    "epe_hands",
    "epe_penalty_sum_px",
    "ct_hands",
    "ct_penalty_sum_m",
    "wrist_penalty_sum_m",
    "go_hands",
    "go_penalty_sum_deg",
    "matched_hands",
    "matched_mpjpe_sum_mm",
    "jitter_sum_mm",
    "jitter_count",
    "all_hand_frames",
    "all_mpjpe_sum_mm",
    "iv_hand_frames",
    "iv_mpjpe_sum_mm",
    "oos_hand_frames",
    "oos_mpjpe_sum_mm",
)


@dataclass
class EvaluationAccumulator:
    values: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in STATE_KEYS})

    def add(self, key: str, value: float | int | Tensor) -> None:
        if key not in self.values:
            raise KeyError(key)
        self.values[key] += float(value)

    def as_tensor(self, device: torch.device) -> Tensor:
        return torch.tensor(
            [self.values[key] for key in STATE_KEYS], device=device, dtype=torch.float64
        )

    def load_tensor(self, value: Tensor) -> None:
        for key, item in zip(STATE_KEYS, value.detach().cpu().tolist()):
            self.values[key] = float(item)

    def finalize(self) -> dict[str, float | int | None]:
        value = self.values

        def ratio(numerator: str, denominator: str) -> float | None:
            count = value[denominator]
            return value[numerator] / count if count else None

        precision_denominator = value["true_positive"] + value["false_positive"]
        recall_denominator = value["true_positive"] + value["false_negative"]
        precision = value["true_positive"] / precision_denominator if precision_denominator else 0.0
        recall = value["true_positive"] / recall_denominator if recall_denominator else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "segments": int(value["segments"]),
            "frames": int(value["frames"]),
            "true_positive": int(value["true_positive"]),
            "false_positive": int(value["false_positive"]),
            "false_negative": int(value["false_negative"]),
            "FAcc": ratio("correct_frames", "frames"),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MPJPE-p_mm": ratio("mpjpe_penalty_sum_mm", "penalty_hands"),
            "PA-p_mm": ratio("pa_penalty_sum_mm", "penalty_hands"),
            "EPE2D-p_px": ratio("epe_penalty_sum_px", "epe_hands"),
            "GO-p_deg": ratio("go_penalty_sum_deg", "go_hands"),
            "CT-p_m": ratio("ct_penalty_sum_m", "ct_hands"),
            "Wrist-p_m": ratio("wrist_penalty_sum_m", "ct_hands"),
            "Jitter_mm_per_frame2": ratio("jitter_sum_mm", "jitter_count"),
            "MPJPE-matched_mm": ratio("matched_mpjpe_sum_mm", "matched_hands"),
            "MPJPE-IV_mm": ratio("iv_mpjpe_sum_mm", "iv_hand_frames"),
            "MPJPE-OOS_mm": ratio("oos_mpjpe_sum_mm", "oos_hand_frames"),
            "MPJPE+OOS_mm": ratio("all_mpjpe_sum_mm", "all_hand_frames"),
            "IV_hand_frames": int(value["iv_hand_frames"]),
            "OOS_hand_frames": int(value["oos_hand_frames"]),
        }


def _camera_kwargs(batch: Mapping[str, object]) -> dict[str, object]:
    return {
        "distortion": batch["distortion"],
        "camera_model": batch["camera_model"],
        "camera_parameters": batch["camera_parameters"],
        "source_image_size": batch["source_image_size"],
    }


def _selected_sum(value: Tensor, mask: Tensor, scale: float = 1.0) -> float:
    return float(torch.where(mask, value, torch.zeros_like(value)).sum().double().cpu()) * scale


@torch.no_grad()
def score_batch(
    accumulator: EvaluationAccumulator,
    output: DreamHandOutput,
    batch: Mapping[str, object],
    gt_vertices_root: Tensor,
    canonical_joints: Tensor,
    *,
    solver: str,
    gt_mano_translation: Tensor,
) -> None:
    """Accumulate detection, pose and motion metrics for one dataset batch.

    Ground-truth calibration is used for scoring both camera modes; the
    K-free model branch ignores it. Standard EPE scores anchors; K-free EPE
    scores the projected 3D result. Architecture does not change this protocol.
    """

    from .data.policy import require_dataset

    require_dataset(str(batch["dataset"]))
    intrinsics = batch["intrinsics"]
    image_size = batch["image_size"]
    if not isinstance(intrinsics, Tensor) or not isinstance(image_size, Tensor):
        raise TypeError("camera tensors are missing")
    camera = _camera_kwargs(batch)
    gt_joints_camera = batch["joints_camera"]
    gt_joints_root = batch["joints_root"]
    valid_hand = batch["valid_hand"].bool()
    valid_mano = batch["valid_mano"].bool()
    valid_3d = batch["valid_joints_3d"].bool() & valid_hand.unsqueeze(-1)
    valid_2d = batch["valid_joints_2d"].bool() & valid_hand.unsqueeze(-1)
    if not isinstance(gt_joints_camera, Tensor) or not isinstance(gt_joints_root, Tensor):
        raise TypeError("joint tensors are missing")

    gt_projected = project_camera(
        gt_joints_camera,
        intrinsics,
        image_size,
        **camera,  # type: ignore[arg-type]
    )
    pred_projected = project_camera(
        output.joints_camera,
        intrinsics,
        image_size,
        **camera,  # type: ignore[arg-type]
    )
    active = output.decoder.existence_logits.sigmoid() > 0.5

    gt_on_screen = valid_hand & on_screen_from_projection(gt_joints_camera, gt_projected, valid_3d)
    pred_on_screen = on_screen_from_projection(output.joints_camera, pred_projected)
    candidate = active & pred_on_screen
    gt_box_points = gt_vertices_root + batch["translation"].unsqueeze(-2)
    pred_box_points = output.vertices_camera
    gt_boxes = projected_boxes(
        gt_box_points,
        intrinsics,
        image_size,
        valid=None,
        **camera,  # type: ignore[arg-type]
    )
    pred_boxes = projected_boxes(
        pred_box_points,
        intrinsics,
        image_size,
        valid=None,
        **camera,  # type: ignore[arg-type]
    )
    overlap = aligned_iou(pred_boxes, dilate_boxes(gt_boxes, 0.10))
    matched = gt_on_screen & candidate & torch.isfinite(overlap) & (overlap > 0.0)
    false_negative = gt_on_screen & ~matched
    false_positive = candidate & ~matched
    detection_matched = matched
    detection_false_negative = false_negative
    detection_false_positive = false_positive

    accumulator.add("segments", gt_joints_camera.shape[0])
    accumulator.add("frames", gt_joints_camera.shape[0] * gt_joints_camera.shape[1])
    accumulator.add(
        "correct_frames",
        (~(detection_false_negative | detection_false_positive).any(-1)).sum(),
    )
    accumulator.add("true_positive", detection_matched.sum())
    accumulator.add("false_positive", detection_false_positive.sum())
    accumulator.add("false_negative", detection_false_negative.sum())
    accumulator.add("penalty_hands", gt_on_screen.sum())

    canonical = canonical_joints.to(gt_joints_root).view(1, 1, 2, 21, 3).expand_as(gt_joints_root)
    predicted_root = output.joints_root_mano
    mpjpe = masked_wrist_aligned_mpjpe(predicted_root.float(), gt_joints_root.float(), valid_3d)
    canonical_mpjpe = masked_wrist_aligned_mpjpe(
        canonical.float(), gt_joints_root.float(), valid_3d
    )
    pa = masked_procrustes_mpjpe(predicted_root.float(), gt_joints_root.float(), valid_3d)
    canonical_pa = masked_procrustes_mpjpe(canonical.float(), gt_joints_root.float(), valid_3d)
    accumulator.add(
        "mpjpe_penalty_sum_mm",
        _selected_sum(mpjpe, matched, 1000.0)
        + _selected_sum(canonical_mpjpe, false_negative, 1000.0),
    )
    accumulator.add(
        "pa_penalty_sum_mm",
        _selected_sum(pa, matched, 1000.0) + _selected_sum(canonical_pa, false_negative, 1000.0),
    )
    accumulator.add("matched_hands", matched.sum())
    accumulator.add("matched_mpjpe_sum_mm", _selected_sum(mpjpe, matched, 1000.0))

    if solver == "standard":
        predicted_2d = output.decoder.anchors_2d.float()
    elif solver == "kfree":
        predicted_2d = pred_projected
    else:
        raise ValueError("solver must be standard or kfree")
    pixel_scale = torch.stack((image_size[:, 1], image_size[:, 0]), dim=-1)
    pixel_scale = pixel_scale[:, None, None, None]
    joint_epe = ((predicted_2d - batch["joints_2d"]) * pixel_scale).norm(dim=-1)
    visible_count = valid_2d.sum(-1).clamp_min(1)
    epe = (joint_epe * valid_2d).sum(-1) / visible_count
    diagonal = image_size.square().sum(-1).sqrt()[:, None, None].expand_as(epe)
    epe_population = gt_on_screen & valid_2d.any(-1)
    epe_matched = matched & epe_population
    epe_missed = false_negative & epe_population
    accumulator.add("epe_hands", epe_population.sum())
    accumulator.add(
        "epe_penalty_sum_px",
        _selected_sum(epe, epe_matched) + _selected_sum(diagonal, epe_missed),
    )

    ct = (output.mano_translation.float() - gt_mano_translation.float()).norm(dim=-1)
    canonical_ct = gt_mano_translation.float().norm(dim=-1)
    wrist = (output.pnp.translation.float() - batch["translation"].float()).norm(dim=-1)
    ct_population = gt_on_screen & valid_mano
    ct_matched = matched & ct_population
    ct_missed = false_negative & ct_population
    accumulator.add("ct_hands", ct_population.sum())
    accumulator.add(
        "ct_penalty_sum_m",
        _selected_sum(ct, ct_matched) + _selected_sum(canonical_ct, ct_missed),
    )
    accumulator.add(
        "wrist_penalty_sum_m",
        _selected_sum(wrist, ct_matched)
        + _selected_sum(batch["translation"].float().norm(dim=-1), ct_missed),
    )
    go_population = gt_on_screen & valid_mano
    go_matched = matched & valid_mano
    go_missed = false_negative & valid_mano
    go = global_orientation_error(
        output.decoder.global_rotation.float(), batch["global_rotation"].float()
    )
    identity = torch.eye(3, device=go.device).expand_as(batch["global_rotation"])
    canonical_go = global_orientation_error(identity, batch["global_rotation"].float())
    accumulator.add("go_hands", go_population.sum())
    accumulator.add(
        "go_penalty_sum_deg",
        _selected_sum(go, go_matched) + _selected_sum(canonical_go, go_missed),
    )

    all_hand = valid_hand
    oos = all_hand & ~gt_on_screen
    iv = all_hand & gt_on_screen
    accumulator.add("all_hand_frames", all_hand.sum())
    accumulator.add("all_mpjpe_sum_mm", _selected_sum(mpjpe, all_hand, 1000.0))
    accumulator.add("iv_hand_frames", iv.sum())
    accumulator.add("iv_mpjpe_sum_mm", _selected_sum(mpjpe, iv, 1000.0))
    accumulator.add("oos_hand_frames", oos.sum())
    accumulator.add("oos_mpjpe_sum_mm", _selected_sum(mpjpe, oos, 1000.0))

    for batch_index in range(matched.shape[0]):
        for side in range(2):
            jitter_sum, jitter_count = jitter_matched_run_sum_count(
                output.joints_camera[batch_index, :, side].float(),
                matched[batch_index, :, side],
            )
            accumulator.add("jitter_sum_mm", float(jitter_sum.cpu()) * 1000.0)
            accumulator.add("jitter_count", jitter_count)


def validate_metric_result(result: Mapping[str, float | int | None]) -> None:
    """Fail a long evaluation if a reported numeric metric is not finite."""

    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite evaluation result {key}={value}")
