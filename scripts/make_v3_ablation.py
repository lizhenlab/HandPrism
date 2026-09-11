#!/usr/bin/env python3
"""Generate explicit, independently switchable Fusion experiments; never train."""
from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path

from scripts.train import load_config, validate_config
from handprism.architectures import FUSION, add_architecture_argument
from handprism.config import LossWeights
from handprism.fusion import FusionConfig


PRESETS = {
    "B0": (),
    "edge": ("edge",),
    "M1": ("readout",),
    "M2": ("local-rgb",),
    "M3": ("joint-mano", "mano-loss", "consistency"),
    "M4": ("wrist-prior", "depth-loss"),
    "M5": ("quality", "irls"),
    "M6": ("motion", "augmentation", "hard-sampling", "loss-rebalance"),
}
FEATURES = tuple(dict.fromkeys(item for values in PRESETS.values() for item in values))
PRESETS["all"] = FEATURES


def build_ablation(base: dict, features: tuple[str, ...], *, clip_budget: int | None = None,
                   world_size: int = 8) -> dict:
    """Each preset is B0 + named changes, NOT implicitly cumulative.

    Features separately test MANO GT loss vs refinement, uncertainty vs IRLS,
    and motion vs appearance vs sampling. Keep split, assets, batch and seed.
    """
    if base["architecture"] != FUSION or set(features) - set(FEATURES):
        raise ValueError("unknown Fusion ablation feature/architecture")
    if "consistency" in features and "mano-loss" not in features:
        raise ValueError("consistency requires an explicit mano-loss feature")
    if "wrist-prior" in features and "depth-loss" not in features:
        raise ValueError("wrist-prior requires an explicit depth-loss feature")
    config = copy.deepcopy(base)
    config["fusion"] = asdict(FusionConfig.from_dict(config["fusion"]))
    for key in ("final_readout", "local_rgb", "joint_mano", "temporal_wrist", "reliability", "edge_quality"):
        config["fusion"][key] = False
    config["decoder"]["anchor_offset_cells"] = 0.
    config["geometry_refinement"] = dict(robust_iterations=0, robust_huber_px=8., depth_refine_fraction=0.)
    config["hard_window_fraction"] = 0.
    config["augmentation"] = {**config.get("augmentation", {}), "enabled": False}
    weights = asdict(LossWeights())
    for feature, flag in (("readout", "final_readout"), ("local-rgb", "local_rgb"),
                           ("joint-mano", "joint_mano"), ("wrist-prior", "temporal_wrist"),
                           ("quality", "reliability"), ("edge", "edge_quality")):
        config["fusion"][flag] = feature in features
    if "edge" in features:
        config["decoder"]["anchor_offset_cells"] = 1.
    if "mano-loss" in features:
        weights["joints_root_mano"] = 5.
    if "consistency" in features:
        weights["direct_mano_consistency"] = .1
    if "wrist-prior" in features:
        weights["wrist_prior"] = .1
    if "depth-loss" in features:
        weights["log_depth"] = .5
    if "quality" in features:
        weights["reliability"] = .1
    if "irls" in features:
        config["geometry_refinement"]["robust_iterations"] = 2
    if "motion" in features:
        weights.update(acceleration=0., velocity_error=.02, acceleration_error=.002,
                       wrist_velocity_error=.01, wrist_acceleration_error=.001)
    if "augmentation" in features:
        config["augmentation"]["enabled"] = True
    if "hard-sampling" in features:
        config["hard_window_fraction"] = .35
    if "loss-rebalance" in features:
        weights.update(rotation_matrix=.25, joints_root=5., joints_camera=2., wrist=.5,
                       reprojection_joints=.5, reprojection_wrist=0., translation=0., acceleration=0.)
    config["loss_weights"] = weights
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be positive")
    batch = world_size * config["batch_size_per_gpu"] * config["gradient_accumulation"]
    if clip_budget is not None:
        if type(clip_budget) is not int or clip_budget < batch or clip_budget % batch:
            raise ValueError("clip budget must be an exact positive multiple of global batch")
        config["steps"] = clip_budget // batch
    config["ablation"] = {
        "features": list(features), "world_size": world_size, "global_batch": batch,
        "global_clips": config["steps"] * batch, "global_frames": config["steps"] * batch * 81,
        "initialization": "shared global and ray weights retain the same seeded initialization",
    }
    config["ablation_note"] = "Candidate only; full validation chooses checkpoints, never test data"
    return validate_config(config, architecture=FUSION)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_architecture_argument(parser)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stage", choices=tuple(PRESETS), help="isolated B0 + preset; not cumulative")
    group.add_argument("--features", nargs="+", choices=FEATURES, help="explicit combination")
    parser.add_argument("--clip-budget", type=int, help="equalize exposure, not just optimizer steps")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--anchor-offset-cells", type=float, choices=(0., .5, 1.))
    parser.add_argument("--fit-target", choices=("pinhole_compatible", "effective_camera", "raw_bearings"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root / "configs"):
        raise ValueError("ablation configs must stay in this workspace's configs/")
    config = load_config(args.base, architecture=args.architecture)
    if args.architecture != FUSION:
        raise ValueError("these ablation options belong to HandPrism-Fusion")
    if args.stage is not None or args.features is not None:
        config = build_ablation(config, PRESETS[args.stage] if args.stage else tuple(args.features),
                                clip_budget=args.clip_budget, world_size=args.world_size)
    elif args.clip_budget is not None:
        raise ValueError("clip-budget requires an explicit stage or feature set")
    if args.anchor_offset_cells is not None:
        config["decoder"]["anchor_offset_cells"] = args.anchor_offset_cells
    if args.fit_target is not None:
        if config["solver"] != "kfree":
            raise ValueError("camera-fit targets are only valid for K-free")
        config["kfree_camera_fit"]["target"] = args.fit_target
    config["name"] = output.stem
    validate_config(config, architecture=FUSION)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(config, indent=2, allow_nan=False) + "\n")
    print(f"Created {output}; no training started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
