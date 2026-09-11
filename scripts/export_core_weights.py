#!/usr/bin/env python3
"""Export one verified final Core checkpoint without optimizer or private metadata."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from handprism.architectures import CORE, require_legacy_digest, validate_checkpoint_identity
from handprism.data.policy import allowed_path
from handprism.release import RELEASE_FORMAT, file_sha256, inference_settings
from scripts.train import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, destination = allowed_path(args.checkpoint), allowed_path(args.output)
    if destination.exists() or destination.suffix != ".safetensors":
        raise ValueError("output must be a new .safetensors file")
    config = load_config(allowed_path(args.config), architecture=CORE)
    digest = file_sha256(source)
    require_legacy_digest(CORE, config["solver"], digest)
    permitted = {"numpy.ndarray", "numpy._core.multiarray._reconstruct", "numpy.dtype"}
    unknown = set(torch.serialization.get_unsafe_globals_in_checkpoint(source)) - permitted
    if unknown:
        raise ValueError(f"unexpected checkpoint globals: {sorted(unknown)}")
    # These pinned checkpoints contain a NumPy uint32 RNG array. Permit only
    # its standard type constructors; weights_only remains enabled throughout.
    with torch.serialization.safe_globals([
        np.ndarray, np.dtype, np._core.multiarray._reconstruct, type(np.dtype("uint32")),
    ]):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    validate_checkpoint_identity(checkpoint, config, architecture=CORE, legacy_sha256=digest)
    tensors = checkpoint["trainable"]
    allowed_mano_parameters = {"hand.mano.left.body_pose", "hand.mano.right.body_pose"}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"invalid trainable tensor: {name}")
        if name.startswith("hand.mano."):
            if name not in allowed_mano_parameters or tuple(value.shape) != (1, 3):
                raise ValueError("MANO assets must not be exported")
    metadata = {
        "format": RELEASE_FORMAT, "architecture": CORE,
        "implementation_id": config["implementation_id"], "solver": config["solver"],
        "step": "20000", "source_sha256": digest,
        "datasets": json.dumps(["arctic", "hot3d"]),
        "inference_settings": json.dumps(inference_settings(config), sort_keys=True),
        "scope": "trainable tensors only; no optimizer, RNG, dataset or MANO geometry assets",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()},
              str(destination), metadata=metadata)
    exported = load_file(str(destination), device="cpu")
    if exported.keys() != tensors.keys() or any(
        exported[key].dtype != value.dtype or not torch.equal(exported[key], value)
        for key, value in tensors.items()
    ):
        raise RuntimeError("exported tensors failed exact equality verification")
    print(json.dumps({"file": destination.name, "solver": config["solver"],
                      "bytes": destination.stat().st_size, "sha256": file_sha256(destination),
                      "source_sha256": digest, "tensors": len(tensors),
                      "parameters": sum(value.numel() for value in tensors.values()),
                      "exact_tensor_equality": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
