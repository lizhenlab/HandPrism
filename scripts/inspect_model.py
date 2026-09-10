#!/usr/bin/env python3
"""Report measured decoder/ray parameter counts for the selected architecture."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamhand.decoder import DreamHandDecoder
from dreamhand.ray import RayHead
from dreamhand.architectures import add_architecture_argument, architecture_spec


def inspect_architecture(architecture: str) -> dict:
    spec = architecture_spec(architecture)
    decoder = DreamHandDecoder(architecture=architecture)
    ray = RayHead()
    decoder_count = sum(p.numel() for p in decoder.parameters())
    ray_count = sum(p.numel() for p in ray.parameters())
    return {
        "architecture": architecture,
        "implementation_id": spec.implementation_id,
        "query_attention": spec.query_attention,
        "decoder_parameters": decoder_count,
        "ray_head_parameters": ray_count,
        "decoder_and_ray_parameters": decoder_count + ray_count,
        "scope": "Default decoder and ray head only; excludes Wan, LoRA, VAE and MANO",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_architecture_argument(parser)
    args = parser.parse_args()
    print(json.dumps(inspect_architecture(args.architecture), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
