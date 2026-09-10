#!/usr/bin/env python3
"""Create an explicit Fusion ablation config; never launch training or overwrite."""

import argparse
import json
from pathlib import Path

from scripts.train import load_config
from dreamhand.architectures import FUSION, add_architecture_argument


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_architecture_argument(parser)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-offset-cells", type=float, choices=(0.0, 0.5, 1.0))
    parser.add_argument(
        "--fit-target", choices=("pinhole_compatible", "effective_camera", "raw_bearings")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root / "configs"):
        raise ValueError("ablation configs must stay in this workspace's configs/")
    config = load_config(args.base, architecture=args.architecture)
    if args.architecture != FUSION:
        raise ValueError("these ablation options belong to HandPrism-Fusion")
    if args.anchor_offset_cells is not None:
        config["decoder"]["anchor_offset_cells"] = args.anchor_offset_cells
    if args.fit_target is not None:
        if config["solver"] != "kfree":
            raise ValueError("camera-fit targets are only valid for K-free")
        config["kfree_camera_fit"]["target"] = args.fit_target
    config["name"] = output.stem
    config["ablation_note"] = (
        "Independent experimental configuration; validation required before long training"
    )
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(config, indent=2, allow_nan=False) + "\n")
    print(f"Created {output}; no training started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
