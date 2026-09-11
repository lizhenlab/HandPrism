#!/usr/bin/env python3
"""Report measured decoder/ray parameter counts for the selected architecture."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handprism.decoder import HandPrismDecoder
from handprism.ray import RayHead
from handprism.architectures import add_architecture_argument, architecture_spec
from handprism.fusion_runtime import fusion_config_from_json
from handprism.config import DecoderConfig


def inspect_architecture(architecture: str, config: dict | None = None) -> dict:
    spec = architecture_spec(architecture)
    value = config or {"architecture": architecture}
    decoder = HandPrismDecoder(DecoderConfig(**value.get("decoder", {})), architecture=architecture,
                               fusion_config=fusion_config_from_json(value))
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
        "scope": "Configured decoder and ray head only; excludes Wan, LoRA, VAE and MANO",
        "fusion": value.get("fusion", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_architecture_argument(parser)
    parser.add_argument("--config", type=Path, help="Include the selected Fusion modules in parameter counts")
    args = parser.parse_args()
    from scripts.train import load_config
    config = load_config(args.config, architecture=args.architecture) if args.config else None
    print(json.dumps(inspect_architecture(args.architecture, config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
