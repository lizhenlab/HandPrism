"""One configuration path shared by training, evaluation and inference."""
from __future__ import annotations

from dataclasses import fields
import math
import torch

from .architectures import CORE
from .config import LossWeights
from .fusion import FusionConfig
from .data.augmentation import AugmentationConfig


def fusion_config_from_json(config: dict) -> FusionConfig:
    result = FusionConfig.from_dict(config.get("fusion"))
    if config.get("architecture") == CORE and result.enabled:
        raise ValueError("Core does not support Fusion modules")
    return result


def ddp_options(config: dict) -> dict:
    # Non-reentrant Wan checkpointing permits unused-parameter discovery.
    # Avoid static-graph first-backward/no_sync interactions for Fusion's
    # accumulated microbatches; keep the established Core configuration.
    return ({"broadcast_buffers": False, "static_graph": True} if config["architecture"] == CORE
            else {"broadcast_buffers": False, "find_unused_parameters": True})


def loss_weights_from_json(config: dict) -> LossWeights:
    value = config.get("loss_weights", {})
    unknown = set(value) - {f.name for f in fields(LossWeights)}
    if unknown:
        raise ValueError(f"unknown loss weights: {sorted(unknown)}")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in value.values()):
        raise ValueError("loss weights must be finite nonnegative numbers")
    result = LossWeights(**value)
    if config.get("architecture") == CORE and result != LossWeights():
        raise ValueError("Core loss weights are frozen")
    opts = fusion_config_from_json(config)
    if bool(result.reliability) != opts.reliability:
        raise ValueError("reliability head and its supervised loss must be enabled together")
    if bool(result.wrist_prior) != opts.temporal_wrist:
        raise ValueError("independent wrist head requires its uncertainty loss")
    if opts.temporal_wrist and not result.log_depth:
        raise ValueError("independent wrist fallback requires independent log_depth supervision")
    if result.direct_mano_consistency and not result.joints_root_mano:
        raise ValueError("direct/MANO consistency requires MANO GT supervision")
    return result


def dataset_options(config: dict, *, training: bool) -> dict:
    if config["architecture"] == CORE:
        return {}
    fusion = fusion_config_from_json(config)
    augment = config.get("augmentation", {}) if training else {}
    AugmentationConfig(**augment)
    return {
        "extended_contract": True,
        "detail_long_side": fusion.local_source_long_side if fusion.local_rgb else 0,
        "augmentation": augment,
        "hard_window_fraction": config.get("hard_window_fraction", 0.) if training else 0.,
    }


def fusion_forward_options(config: dict, batch: dict, *, training: bool = False, step: int = 0) -> dict:
    if config["architecture"] == CORE:
        return {}
    options = {"rgb_high": batch.get("rgb_high")}
    fusion = fusion_config_from_json(config)
    if fusion.local_rgb and options["rgb_high"] is None:
        raise ValueError("local_rgb requires aligned original-resolution rgb_high, not upscaled video")
    if training:
        options["optimizer_step"] = step
        if fusion.local_rgb and step < fusion.roi_teacher_steps:
            options.update(roi_teacher=batch["joints_2d"], roi_teacher_valid=batch["valid_joints_2d"])
    return options


def validation_selection_score(metrics: dict) -> float:
    """Frozen equal-dataset accuracy/coverage composite; lower is better.

    Fixed engineering scales, not fitted on test. Only FULL validation can
    select checkpoints. Component metrics must always be reported alongside.
    """
    scores = []
    for dataset in ("arctic", "hot3d"):
        prefix = f"val/{dataset}/test_protocol/"
        keys = ("CameraMPJPE_mm", "MPJPE+OOS_mm", "AnchorEPE_px", "ExistenceCoverage", "F1")
        values = [metrics.get(prefix + key) for key in keys]
        if any(v is None or not math.isfinite(float(v)) for v in values):
            raise ValueError(f"full validation lacks finite accuracy/coverage for {dataset}")
        camera, root, anchor, coverage, f1 = values
        scores.append(camera / 100. + root / 50. + anchor / 20. + 5 * (1-coverage) + 5 * (1-f1))
    return sum(scores) / len(scores)


def weighted_loss_audit(terms: dict[str, float], config: dict, step: int) -> dict:
    """Actual signed contributions, not per-task gradient magnitudes.

    Consistency is already ramped in the criterion; camera_fit is not.
    Laplace NLL may legitimately be negative.
    """
    weights = loss_weights_from_json(config)
    weighted = {}
    for name, value in terms.items():
        if name == "total":
            continue
        weight = getattr(weights, name)
        if name == "camera_fit":
            warmup = config.get("kfree_camera_fit", {}).get("warmup_steps", 500)
            weight *= min(max(step / max(warmup, 1), 0.), 1.) if config["solver"] == "kfree" else 0.
        weighted[name] = weight * value
    if not all(math.isfinite(v) for v in (*weighted.values(), terms["total"])):
        raise RuntimeError("non-finite loss audit")
    summed = sum(weighted.values())
    if not math.isclose(summed, terms["total"], rel_tol=1e-4, abs_tol=1e-5):
        raise RuntimeError("weighted loss contributions do not reconstruct total")
    return {"weighted": weighted, "sum": summed}


@torch.no_grad()
def training_diagnostics(output, batch: dict) -> dict[str, torch.Tensor]:
    """Clip-level geometry/ROI diagnostics, never included in the objective."""
    pnp, decoded = output.pnp, output.decoder
    solved = (~pnp.used_fallback).float().mean()
    weight = pnp.geometry_weight
    stats = {"pnp_solved_fraction": solved,
             "geometry_weight_mean": weight.mean() if weight is not None else solved,
             "wrist_prior_fraction": (1-weight).mean() if decoded.wrist_prior is not None else solved * 0}
    if decoded.local_roi_bounds is not None:
        bounds, uv = decoded.local_roi_bounds, batch["joints_2d"]
        valid = (batch["valid_joints_2d"].bool() & batch["valid_hand"][..., None].bool()
                 & torch.isfinite(uv).all(-1))
        inside = ((uv >= bounds[..., None, :2]) & (uv <= bounds[..., None, 2:])).all(-1)
        inside &= decoded.local_roi_valid[..., None]
        stats.update(roi_valid_fraction=decoded.local_roi_valid.float().mean(),
                     roi_joint_coverage=(inside & valid).sum().float() / valid.sum().clamp_min(1),
                     roi_supervised_joints=valid.sum().float())
    return stats


def decoder_gradient_norms(decoder) -> dict[str, torch.Tensor]:
    """Post-accumulation, pre-clipping branch norms; zero is diagnostic data."""
    result = {}
    for name in ("camera_head", "pose_head", "joint_head", "feature_projection", "final_spatial", "refinement"):
        module = getattr(decoder, name, None)
        if module is not None:
            parameters = list(module.parameters())
            zero = parameters[0].new_zeros(())
            result[f"gradient/{name}"] = sum(
                (p.grad.detach().float().square().sum() for p in parameters if p.grad is not None), zero).sqrt()
    return result
