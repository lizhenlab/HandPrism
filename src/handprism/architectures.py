"""Explicit HandPrism architecture identities, independent of camera solver."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

CORE = "handprism-core"
FUSION = "handprism-fusion"
ARCHITECTURES = (CORE, FUSION)
CHECKPOINT_FORMAT = "handprism-training-checkpoint"
CONTRACT_VERSION = 9


@dataclass(frozen=True)
class Architecture:
    name: str
    display_name: str
    implementation_id: str
    query_attention: str
    direct_joints: str
    pnp_residual: str

    @property
    def contract(self) -> dict[str, str]:
        return {
            "query_attention": self.query_attention,
            "direct_joints": self.direct_joints,
            "pnp_residual": self.pnp_residual,
            "backbone_grid": "official_32x_spatial_compression",
        }


_SPECS = {
    CORE: Architecture(
        CORE, "HandPrism-Core", "handprism-core-r1",
        "per_query_bidirectional_temporal_RoPE", "hidden_feature_interpolation",
        "bearing_residual_times_image_diagonal_proxy",
    ),
    FUSION: Architecture(
        FUSION, "HandPrism-Fusion", "handprism-fusion-r4",
        "bidirectional_joint_time_query_with_temporal_RoPE", "latent_coordinate_interpolation",
        "camera_pixel_projection_or_predicted_ray_inverse",
    ),
}

# Only these two preserved final checkpoints can enter the legacy inference
# path. No architecture is guessed from tensor shapes or a filename.
LEGACY_CORE_SHA256 = {
    "standard": "685da372b8ed631d23d720250a541c43a59816c3927d50b81c64ae5ba27f40f7",
    "kfree": "821a2684f5ae9f4fab7c8329123de0e1e30e43f8317f4efa95a0c574bd1698e4",
}


def architecture_spec(name: str) -> Architecture:
    if name not in _SPECS:
        raise ValueError(f"architecture must be explicitly selected from {ARCHITECTURES}, got {name!r}")
    return _SPECS[name]


def add_architecture_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--architecture", required=True, choices=ARCHITECTURES,
        help="required model architecture; independent of standard/kfree camera mode",
    )


def require_config_architecture(config: dict[str, Any], architecture: str) -> Architecture:
    spec = architecture_spec(architecture)
    if config.get("architecture") != architecture:
        raise ValueError("CLI architecture and configuration architecture disagree")
    if config.get("implementation_id") != spec.implementation_id:
        raise ValueError("configuration implementation_id does not match architecture")
    if config.get("architecture_contract") != spec.contract:
        raise ValueError("configuration architecture_contract does not match selected model")
    return spec


def require_legacy_digest(architecture: str, solver: str, digest: str) -> None:
    if architecture != CORE or digest != LEGACY_CORE_SHA256.get(solver):
        raise ValueError("legacy weights require HandPrism-Core and a preserved final SHA-256")


def validate_checkpoint_identity(
    checkpoint: dict[str, Any], config: dict[str, Any], *, architecture: str,
    legacy_sha256: str | None = None,
) -> None:
    spec = require_config_architecture(config, architecture)
    if legacy_sha256 is not None:
        require_legacy_digest(architecture, config["solver"], legacy_sha256)
        old = checkpoint.get("config", {})
        if (checkpoint.get("step") != 20000 or old.get("solver") != config["solver"]
                or old.get("dataset_weights") != {"arctic": 0.4375, "hot3d": 0.5625}
                or set(old.get("dataset_roots", {})) != {"arctic", "hot3d"}):
            raise ValueError("legacy checkpoint solver/data contract mismatch")
        if config.get("decoder", {}).get("anchor_offset_cells") != 0:
            raise ValueError("legacy Core weights do not contain an anchor offset head")
        return
    if (checkpoint.get("format") != CHECKPOINT_FORMAT
            or checkpoint.get("architecture") != architecture
            or checkpoint.get("implementation_id") != spec.implementation_id):
        raise ValueError("checkpoint architecture/implementation mismatch; no automatic fallback")
    if checkpoint.get("config") != config:
        raise ValueError("checkpoint config differs from the requested run config")
