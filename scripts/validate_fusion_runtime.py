#!/usr/bin/env python3
"""Bounded Fusion validation, never a full training job or benchmark result.

Modes: real-data decodes five TRAIN frames/source; heads checks two synthetic
updates with real MANO (optionally DDP); wan includes real VAE/DiT weights on a
5x64x64 synthetic clip. No test RGB, trained weights or old run files are used.
"""
from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

import argparse
import copy
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from handprism.architectures import FUSION, add_architecture_argument
from handprism.backbone import (WanCleanLatentEncoder, WanFrozenVAEEncoder,
                               load_official_vae, load_official_wan)
from handprism.camera import project_camera
from handprism.completion import require_finite_json
from handprism.config import DecoderConfig
from handprism.data.arctic import load_arctic_window
from handprism.data.hot3d import load_hot3d_window
from handprism.data.dataset import read_jsonl
from handprism.data.policy import allowed_path
from handprism.fusion_runtime import (fusion_config_from_json, loss_weights_from_json,
                                     weighted_loss_audit, ddp_options)
from handprism.lora import inject_wan_lora, configure_trainable_backbone, promote_trainable_parameters
from handprism.losses import HandPrismLoss
from handprism.mano import SmplxMano
from handprism.model import HandPrismModel
from handprism.precision import _tensors
from handprism.system import HandPrismSystem
from handprism.training import prediction_from_output, target_from_batch
from scripts.train import load_config, resolve, solver_config_from_json, load_trainable_state, trainable_state


def synthetic_batch(mano, device, frames=5):
    rot = torch.eye(3, device=device).expand(1, frames, 2, 3, 3).clone()
    art = torch.eye(3, device=device).expand(1, frames, 2, 15, 3, 3).clone()
    betas = torch.zeros(1, frames, 2, 10, device=device)
    with torch.no_grad():
        joints, _ = mano(rot, art, betas)
    wrist = torch.tensor([.03, -.02, .7], device=device).expand(1, frames, 2, 3).clone()
    camera = joints + wrist[..., None, :]
    k = torch.tensor([[[60., 0, 32], [0, 60., 32], [0, 0, 1]]], device=device)
    size = torch.tensor([[64., 64.]], device=device)
    uv = project_camera(camera, k, size)
    hand = torch.ones(1, frames, 2, dtype=torch.bool, device=device)
    valid = hand[..., None].expand(-1, -1, -1, 21)
    return dict(dataset="arctic", intrinsics=k, image_size=size, distortion=torch.zeros(1, 8, device=device),
        camera_model="pinhole", camera_parameters=None, source_image_size=None, gt_ray_field=None,
        global_rotation=rot, articulation=art, betas=betas, joints_root=joints, joints_camera=camera,
        translation=wrist, joints_2d=uv, existence=hand, visibility=hand, valid_hand=hand,
        valid_mano=hand, valid_joints_3d=valid, valid_joints_2d=valid,
        valid_ray=torch.ones(1, dtype=torch.bool, device=device),
        timestamps=torch.arange(frames, dtype=torch.float64, device=device)[None]/30,
        in_frame=valid, visibility_valid=hand, synthetic_occluded=torch.zeros_like(valid),
        rgb_high=torch.randint(256, (1, 3, frames, 128, 128), dtype=torch.uint8, device=device))


def real_data(root, config, source):
    report = {}
    for dataset in ("arctic", "hot3d"):
        record = read_jsonl(allowed_path(source) / f"{dataset}_train.jsonl")[0]
        start = int(record["valid_ranges"][0][0] if "valid_ranges" in record else record["start_min"])
        kwargs = dict(mano_model_path=resolve(root, config["mano_model"]), extended_contract=True,
                      detail_long_side=1408)
        sample = (load_arctic_window(record["root"], record["sequence"], start, 5,
                                    image_offset=record["image_offset"], **kwargs) if dataset == "arctic" else
                  load_hot3d_window(record["recording_root"], start, 5,
                                   required_masks=tuple(record["required_masks"]), **kwargs))
        if not all(torch.isfinite(x).all() for x in _tensors(sample) if x.is_floating_point()):
            raise RuntimeError("nonfinite real sample")
        times = sample.timestamps
        report[dataset] = dict(recording=sample.recording_id, frames=5,
            timestamp_source=sample.timestamp_source, delta_seconds=times.diff().tolist(),
            video_shape=list(sample.video.shape), native_detail_shape=list(sample.rgb_high.shape),
            observed_label_count=int(sample.observed_valid.sum()),
            in_frame_count=int(sample.in_frame.sum()), valid_3d_count=int(sample.valid_joints_3d.sum()))
    return report


def gradient_reachability(model, mode, options):
    """Require gradients beyond zero-initialized residuals on update two."""
    prefix = "hand.decoder." if mode == "wan" else "decoder."
    selected = {"global_decoder": prefix + "feature_projection.weight",
                "depth_head": prefix + "camera_head.2.weight"}
    for flag, label, suffix in (
        (options.final_readout, "final_readout", "final_spatial.q.weight"),
        (options.local_rgb, "native_rgb_encoder", "refinement.local.encoder.0.weight"),
        (options.joint_mano, "joint_mano", "refinement.joint_pose.2.weight"),
        (options.temporal_wrist, "wrist_prior", "refinement.wrist.2.weight"),
        (options.reliability, "joint_quality", "refinement.quality.weight"),
    ):
        if flag:
            selected[label] = prefix + suffix
    if mode == "wan":
        selected.update(
            patch_embedding="encoder.model.patch_embedding.weight",
            lora_block_0="encoder.model.blocks.0.self_attn.q.lora_B",
            lora_block_15="encoder.model.blocks.15.self_attn.q.lora_B",
        )
    parameters = dict(model.named_parameters())
    norms = {}
    for label, name in selected.items():
        gradient = parameters[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"missing/nonfinite gradient in {name}")
        norms[label] = float(gradient.detach().double().norm())
        if norms[label] <= 0:
            raise RuntimeError(f"zero gradient in active branch {name}")
    if mode == "wan":
        # The feature tap intentionally does not execute later Wan blocks/head.
        for name in ("encoder.model.blocks.16.self_attn.q.lora_B",):
            if parameters[name].grad is not None:
                raise RuntimeError(f"unexpected gradient beyond the block-15 tap: {name}")
    return norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_architecture_argument(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("real-data", "heads", "wan"), required=True)
    parser.add_argument("--source-manifests", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, help="new report inside this workspace's runs/")
    parser.add_argument("--gradient-audit", action="store_true", help="per-loss decoder gradient norms (heads only)")
    parser.add_argument("--accumulation", type=int, default=1, choices=(1, 2), help="bounded no_sync accumulation check")
    args = parser.parse_args()
    if args.architecture != FUSION:
        raise ValueError("this diagnostic cannot alter or train Core")
    root = Path(__file__).resolve().parents[1]
    config = load_config(args.config, architecture=FUSION)
    destination = None
    if args.output is not None:
        from handprism.paths import v3_run_path
        destination = v3_run_path(root, args.output)
        if destination.exists():
            raise FileExistsError(destination)
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1 and args.mode != "heads":
        raise ValueError("only bounded heads diagnostics support DDP")
    torch.set_num_threads(1)
    started = time.perf_counter()
    if args.mode == "real-data":
        if args.source_manifests is None:
            raise ValueError("real-data requires explicit source-manifests")
        result = real_data(root, config, args.source_manifests)
    else:
        device = torch.device(args.device)
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", device.index or 0)))
            torch.cuda.set_device(device)
        if world > 1:
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        torch.manual_seed(1701)
        mano = SmplxMano(resolve(root, config["mano_model"]), flat_hand_mean=True).to(device)
        options = replace(fusion_config_from_json(config), local_resolution=32, local_chunk_size=4)
        decoder = DecoderConfig(anchor_offset_cells=config["decoder"]["anchor_offset_cells"])
        criterion = HandPrismLoss(loss_weights_from_json(config), options)
        if args.mode == "wan":
            model_dir, videox = resolve(root, config["model_dir"]), resolve(root, config["videox_fun"])
            vae = WanFrozenVAEEncoder(load_official_vae(model_dir/"Wan2.2_VAE.pth", videox, torch_dtype=torch.bfloat16).to(device))
            video = torch.linspace(-1, 1, 3*5*64*64, device=device, dtype=torch.bfloat16).reshape(1, 3, 5, 64, 64)
            with torch.no_grad():
                features = vae(video).detach()
            del vae, video
            wan = load_official_wan(model_dir, videox, torch_dtype=torch.bfloat16)
            inject_wan_lora(wan)
            configure_trainable_backbone(wan)
            promote_trainable_parameters(wan, torch.float32)
            model = HandPrismSystem(WanCleanLatentEncoder(wan, gradient_checkpointing=True), mano,
                decoder, solver_config_from_json(config), architecture=FUSION, fusion_config=options).to(device)
        else:
            model = HandPrismModel(mano, decoder, solver_config_from_json(config),
                                   architecture=FUSION, fusion_config=options).to(device)
            features = torch.randn(1, 2, 4, 4, 3072, device=device)
        torch.manual_seed(2100 + rank)
        batch = synthetic_batch(mano, device)
        features = features + torch.zeros_like(features)  # do not cache trainable Wan features
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type=="cuda" else None,
                                          **ddp_options(config)) if world > 1 else model
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        kwargs = dict(target_frames=5, solver=config["solver"],
            intrinsics=batch["intrinsics"] if config["solver"]=="standard" else None,
            image_size=batch["image_size"], rgb_high=batch["rgb_high"], optimizer_step=2500)
        losses_seen, gradient_audit, active_gradients = [], {}, {}
        for iteration in range(2):
            wrapped.train()
            with torch.autocast(device.type, dtype=torch.bfloat16):
                output = (model if args.accumulation == 2 else wrapped)(features, **kwargs)
                target = target_from_batch(batch, *output.ray_field.shape[1:3])
                losses = criterion(prediction_from_output(output), target, batch["intrinsics"], batch["image_size"],
                                   solver=config["solver"], optimizer_step=2500,
                                   camera_fit_config=solver_config_from_json(config))
            if not all(torch.isfinite(t).all() for t in _tensors((output, losses)) if t.is_floating_point()):
                raise RuntimeError("nonfinite output/loss")
            if args.gradient_audit and iteration == 1:
                if args.mode != "heads" or world != 1:
                    raise ValueError("gradient-audit requires single-process heads mode")
                parameters = [p for n,p in model.named_parameters() if n.startswith("decoder.") and p.requires_grad]
                weights = loss_weights_from_json(config)
                for key, loss in losses.items():
                    if key == "total" or not getattr(weights, key):
                        continue
                    grads = torch.autograd.grad(getattr(weights, key)*loss, parameters, retain_graph=True, allow_unused=True)
                    energy = sum((g.detach().double().square().sum() for g in grads if g is not None),
                                 torch.zeros((), device=device, dtype=torch.float64))
                    gradient_audit[key] = float(energy.sqrt())
            if args.accumulation == 2:
                # First microbatch follows production's no_sync path, including
                # the very first backward after DDP construction.
                with (wrapped.no_sync() if world > 1 else nullcontext()):
                    with torch.autocast(device.type, dtype=torch.bfloat16):
                        micro_output = wrapped(features, **kwargs)
                        micro_losses = criterion(prediction_from_output(micro_output), target,
                            batch["intrinsics"], batch["image_size"], solver=config["solver"],
                            optimizer_step=2500, camera_fit_config=solver_config_from_json(config))
                    (micro_losses["total"] / 2).backward()
                # The synchronized forward must be AFTER no_sync backward.
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    output = wrapped(features, **kwargs)
                    losses = criterion(prediction_from_output(output), target,
                        batch["intrinsics"], batch["image_size"], solver=config["solver"],
                        optimizer_step=2500, camera_fit_config=solver_config_from_json(config))
            (losses["total"] / args.accumulation).backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError("nonfinite/missing gradients")
            if iteration == 1:
                active_gradients = gradient_reachability(model, args.mode, options)
            if world > 1:
                summary = torch.stack([sum(g.detach().double().sum() for g in grads),
                                       sum(g.detach().double().square().sum() for g in grads)])
                gathered = [torch.empty_like(summary) for _ in range(world)]
                dist.all_gather(gathered, summary)
                if not all(torch.equal(gathered[0], item) for item in gathered):
                    raise RuntimeError("rank gradient checksums differ")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            terms = {key: float(value.detach()) for key,value in losses.items()}
            losses_seen.append(terms["total"])
            weighted_loss_audit(terms, config, 2500)
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            expected = model(features, **kwargs).joints_camera.detach().clone()
        weights = {key: value.clone() for key, value in trainable_state(model).items()}
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        parameter = next(p for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            parameter.add_(.01)
        load_trainable_state(model, weights)
        optimizer.load_state_dict(optimizer_state)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            actual = model(features, **kwargs).joints_camera
        if not torch.equal(expected, actual):
            raise RuntimeError("trainable-state/optimizer roundtrip changed outputs")
        result = dict(losses=losses_seen, updates=2, accumulation=args.accumulation, finite_gradients=True,
                      roundtrip_exact=True, gradient_audit=gradient_audit,
                      active_branch_gradient_norms=active_gradients,
                      decoder_parameters=sum(p.numel() for p in model.hand.decoder.parameters()) if args.mode=="wan" else
                                         sum(p.numel() for p in model.decoder.parameters()),
                      max_cuda_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else None)
    report = dict(architecture=FUSION, implementation_id=config["implementation_id"], mode=args.mode,
                  solver=config["solver"], world_size=world, seconds=time.perf_counter()-started,
                  scope="bounded diagnostic, not full training/evaluation or accuracy evidence", result=result)
    require_finite_json(report)
    if rank == 0:
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x") as handle:
                handle.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
