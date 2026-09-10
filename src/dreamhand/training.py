"""HandPrism optimizer groups, warmup-cosine schedule and tensor adapters."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .config import OptimizerConfig
from .lora import lora_parameters
from .losses import DreamHandPrediction, DreamHandTarget
from .model import DreamHandOutput
from .ray import pinhole_rays


def _unique(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return result


def build_optimizer(
    backbone: nn.Module,
    decoder: nn.Module,
    ray_head: nn.Module,
    config: OptimizerConfig = OptimizerConfig(),
) -> AdamW:
    lora = _unique(lora_parameters(backbone))
    patch = _unique(backbone.patch_embedding.parameters())
    head = list(backbone.head.parameters()) if hasattr(backbone, "head") else []
    readout = _unique([*decoder.parameters(), *ray_head.parameters(), *head])
    all_ids = [id(item) for group in (lora, patch, readout) for item in group]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("optimizer parameter groups overlap")
    return AdamW(
        [
            {"params": lora, "lr": config.lora_lr, "name": "lora"},
            {"params": patch, "lr": config.patch_lr, "name": "patch_embedding"},
            {"params": readout, "lr": config.decoder_lr, "name": "decoder_and_heads"},
        ],
        weight_decay=config.weight_decay,
    )


def warmup_cosine(optimizer: Optimizer, config: OptimizerConfig = OptimizerConfig()) -> LambdaLR:
    def factor(step: int) -> float:
        if step < config.warmup_steps:
            return float(step + 1) / float(max(config.warmup_steps, 1))
        progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, factor)


def trainable_parameter_report(
    backbone: nn.Module,
    decoder: nn.Module,
    ray_head: nn.Module,
    tap_block: int = 15,
) -> dict[str, int]:
    lora_total = sum(parameter.numel() for parameter in lora_parameters(backbone))
    lora_reachable = 0
    for index, block in enumerate(backbone.blocks):
        if index <= tap_block:
            lora_reachable += sum(parameter.numel() for parameter in lora_parameters(block))
    patch = sum(parameter.numel() for parameter in backbone.patch_embedding.parameters())
    diffusion_head = sum(parameter.numel() for parameter in backbone.head.parameters())
    decoder_count = sum(parameter.numel() for parameter in decoder.parameters())
    ray = sum(parameter.numel() for parameter in ray_head.parameters())
    return {
        "lora_registered": lora_total,
        "lora_reachable": lora_reachable,
        "patch_embedding": patch,
        "diffusion_head_registered_unreachable": diffusion_head,
        "decoder_routed": decoder_count,
        "ray_head": ray,
        "optimizer_registered": lora_total + patch + diffusion_head + decoder_count + ray,
        "forward_reachable": lora_reachable + patch + decoder_count + ray,
    }


def prediction_from_output(output: DreamHandOutput) -> DreamHandPrediction:
    return DreamHandPrediction(
        global_rotation=output.decoder.global_rotation,
        articulation=output.decoder.articulation,
        betas=output.decoder.betas,
        joints_root_direct=output.decoder.joints_root_direct,
        joints_root_mano=output.joints_root_mano,
        joints_camera=output.joints_camera,
        translation=output.pnp.translation,
        anchors_2d=output.decoder.anchors_2d,
        existence_logits=output.decoder.existence_logits,
        visibility_logits=output.decoder.visibility_logits,
        ray_field=output.ray_field,
        camera_fit=output.camera_fit,
    )


def target_from_batch(
    batch: dict,
    ray_height: int,
    ray_width: int,
) -> DreamHandTarget:
    intrinsics = batch["intrinsics"]
    image_size = batch["image_size"]
    distortion = batch.get("distortion")
    rays = batch.get("gt_ray_field")
    if rays is None:
        rays = pinhole_rays(
            intrinsics,
            ray_height,
            ray_width,
            image_size,
            distortion,
        )
    elif rays.shape[1:3] != (ray_height, ray_width):
        rays = F.interpolate(
            rays.permute(0, 3, 1, 2),
            size=(ray_height, ray_width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        rays = F.normalize(rays, dim=-1, eps=1e-6)
    return DreamHandTarget(
        global_rotation=batch["global_rotation"],
        articulation=batch["articulation"],
        betas=batch["betas"],
        joints_root=batch["joints_root"],
        joints_camera=batch["joints_camera"],
        translation=batch["translation"],
        joints_2d=batch["joints_2d"],
        existence=batch["existence"],
        visibility=batch["visibility"],
        ray_field=rays,
        valid_hand=batch["valid_hand"],
        valid_mano=batch["valid_mano"],
        valid_joints_3d=batch["valid_joints_3d"],
        valid_joints_2d=batch["valid_joints_2d"],
        valid_ray=batch["valid_ray"].bool().view(-1, 1, 1).expand(rays.shape[:-1]),
    )
