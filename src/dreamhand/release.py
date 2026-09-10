"""SHA-pinned, tensor-only Core weights for inference and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dreamhand.architectures import CORE, LEGACY_CORE_SHA256, require_config_architecture

RELEASE_FORMAT = "handprism-inference-safetensors-v1"
RELEASE_SHA256 = {
    "standard": "1fce71c40d7119c0b1fe0a16662c99f030380dc9928c35268d7aaf7d675d9fd8",
    "kfree": "90f98aac9b5393b9b8f4d26395ae72770f4f30970b4a3ed0bc99812e2020d7c2",
}
INFERENCE_FIELDS = (
    "architecture", "implementation_id", "architecture_contract", "solver",
    "decoder", "kfree_camera_fit", "geometry_dtype", "dtype", "trainable_dtype",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_settings(config: dict[str, Any]) -> dict[str, Any]:
    # Data/model locations and training-only settings may vary across machines.
    # Architecture, solver and numerical inference settings must remain fixed.
    return {key: config.get(key) for key in INFERENCE_FIELDS}


def validate_release_metadata(metadata: dict[str, str], config: dict[str, Any],
                              architecture: str) -> None:
    spec = require_config_architecture(config, architecture)
    solver = config["solver"]
    if (architecture != CORE or metadata.get("format") != RELEASE_FORMAT
            or metadata.get("architecture") != CORE
            or metadata.get("implementation_id") != spec.implementation_id
            or metadata.get("solver") != solver
            or metadata.get("step") != "20000"
            or metadata.get("source_sha256") != LEGACY_CORE_SHA256.get(solver)):
        raise ValueError("released weights require the matching Core architecture and solver")
    if json.loads(metadata.get("datasets", "null")) != ["arctic", "hot3d"]:
        raise ValueError("released weights have an unexpected dataset contract")
    if json.loads(metadata.get("inference_settings", "null")) != inference_settings(config):
        raise ValueError("released weights and requested inference settings differ")


def load_released_weights(path: Path, config: dict[str, Any], *,
                         architecture: str) -> dict[str, Any]:
    """Never deserialize pickle or restore optimizer/RNG/training state."""
    require_config_architecture(config, architecture)
    expected = RELEASE_SHA256.get(config["solver"])
    if architecture != CORE or expected is None or file_sha256(path) != expected:
        raise ValueError("released weights SHA-256 or architecture mismatch")
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    validate_release_metadata(metadata, config, architecture)
    tensors = load_file(str(path), device="cpu")
    return {"format": RELEASE_FORMAT, "step": 20000, "trainable": tensors,
            "release_metadata": metadata}
