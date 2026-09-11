#!/usr/bin/env python3
"""One tiny real-weight VAE + block-15 forward/backward on an A800."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from handprism.backbone import (
    WanCleanLatentEncoder,
    WanFrozenVAEEncoder,
    load_official_vae,
    load_official_wan,
)
from handprism.lora import configure_trainable_backbone, inject_wan_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/Wan2.2-Fun-5B-Control"))
    parser.add_argument("--videox-fun", type=Path, default=Path("third_party/VideoX-Fun"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    torch.manual_seed(260820308)
    vae = load_official_vae(
        args.model_dir / "Wan2.2_VAE.pth", args.videox_fun, torch_dtype=dtype
    ).to(device)
    video = torch.linspace(-1, 1, 3 * 5 * 64 * 64, device=device, dtype=dtype).reshape(
        1, 3, 5, 64, 64
    )
    latent = WanFrozenVAEEncoder(vae)(video)
    vae_shape = tuple(latent.shape)
    del vae, video
    gc.collect()
    torch.cuda.empty_cache()

    backbone = load_official_wan(args.model_dir, args.videox_fun, torch_dtype=dtype)
    lora = inject_wan_lora(backbone)
    configure_trainable_backbone(backbone)
    backbone.to(device)
    encoder = WanCleanLatentEncoder(backbone, gradient_checkpointing=True)
    features = encoder(latent)
    loss = features.float().square().mean()
    loss.backward()

    result = {
        "vae_latent_shape": vae_shape,
        "tap_feature_shape": tuple(features.shape),
        "lora_modules": lora.modules,
        "lora_parameters": lora.parameters,
        "patch_grad_finite": all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in backbone.patch_embedding.parameters()
        ),
        "block0_lora_b_grad": backbone.blocks[0].self_attn.q.lora_B.grad is not None,
        "block15_lora_b_grad": backbone.blocks[15].self_attn.q.lora_B.grad is not None,
        "block16_lora_b_grad": backbone.blocks[16].self_attn.q.lora_B.grad is not None,
        "diffusion_head_grad": any(parameter.grad is not None for parameter in backbone.head.parameters()),
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if vae_shape != (1, 48, 2, 4, 4):
        raise SystemExit("official VAE spatial/temporal shape differs from audited path")
    if tuple(features.shape) != (1, 2, 2, 2, 3072):
        raise SystemExit("official DiT patch shape differs from audited path")
    if lora.modules != 300 or lora.parameters != 161_218_560:
        raise SystemExit("LoRA schema/count mismatch")
    if not result["patch_grad_finite"] or not result["block0_lora_b_grad"]:
        raise SystemExit("reachable trainable path did not receive finite gradients")
    if result["block16_lora_b_grad"] or result["diffusion_head_grad"]:
        raise SystemExit("an inactive backbone path unexpectedly received gradients")


if __name__ == "__main__":
    main()
