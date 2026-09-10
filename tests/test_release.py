"""Public tensor-only weights: digest, identity, settings and entrypoint guards."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from dreamhand.architectures import CORE, FUSION, LEGACY_CORE_SHA256
from dreamhand import release
from scripts.train import load_config, load_trainable_state

ROOT = Path(__file__).resolve().parents[1]


def config_for(solver="standard", architecture=CORE):
    return load_config(ROOT / "configs" / f"{architecture.replace('-', '_')}_{solver}.json",
                       architecture=architecture)


def metadata_for(config):
    return {"format": release.RELEASE_FORMAT, "architecture": CORE,
            "implementation_id": config["implementation_id"], "step": "20000",
            "solver": config["solver"], "source_sha256": LEGACY_CORE_SHA256[config["solver"]],
            "datasets": json.dumps(["arctic", "hot3d"]),
            "inference_settings": json.dumps(release.inference_settings(config))}


@pytest.mark.parametrize("solver", ["standard", "kfree"])
def test_release_safe_roundtrip(tmp_path, monkeypatch, solver):
    config = config_for(solver)
    module = torch.nn.Linear(2, 3)
    tensors = {name: value.detach().clone() for name, value in module.named_parameters()}
    path = tmp_path / "small.safetensors"
    save_file(tensors, str(path), metadata=metadata_for(config))
    monkeypatch.setitem(release.RELEASE_SHA256, solver, release.file_sha256(path))

    def forbid_pickle(*args, **kwargs):
        pytest.fail("released weights must not use torch.load/pickle")

    monkeypatch.setattr(torch, "load", forbid_pickle)
    checkpoint = release.load_released_weights(path, config, architecture=CORE)
    assert checkpoint["step"] == 20000 and "optimizer" not in checkpoint
    load_trainable_state(module, checkpoint["trainable"])
    for name, value in module.named_parameters():
        assert torch.equal(value, tensors[name])
    with pytest.raises(ValueError, match="SHA-256 or architecture"):
        release.load_released_weights(path, config_for(solver, FUSION), architecture=FUSION)


@pytest.mark.parametrize("field,value", [
    ("format", "unknown"), ("architecture", FUSION), ("implementation_id", "wrong"),
    ("solver", "kfree"), ("step", "19999"), ("source_sha256", "unverified"),
    ("datasets", '["arctic", "hot3d", "other"]'), ("inference_settings", "{}"),
])
def test_release_rejects_metadata_mismatch(field, value):
    config = config_for()
    metadata = metadata_for(config)
    metadata[field] = value
    with pytest.raises(ValueError):
        release.validate_release_metadata(metadata, config, CORE)


def test_release_allows_paths_but_not_geometry_changes():
    config = config_for()
    metadata = metadata_for(config)
    relocated = copy.deepcopy(config)
    relocated["model_dir"] = "models/custom-location"
    relocated["dataset_roots"] = {"arctic": "data/a", "hot3d": "data/h"}
    release.validate_release_metadata(metadata, relocated, CORE)
    relocated["decoder"]["anchor_offset_cells"] = 0.1
    with pytest.raises(ValueError, match="inference settings"):
        release.validate_release_metadata(metadata, relocated, CORE)


def test_release_digest_is_required_before_loading(tmp_path):
    path = tmp_path / "wrong.safetensors"
    path.write_bytes(b"not a weights file")
    with pytest.raises(ValueError, match="SHA-256"):
        release.load_released_weights(path, config_for(), architecture=CORE)


@pytest.mark.parametrize("entrypoint", ["infer.py", "evaluate.py"])
def test_release_flag_conflicts_with_legacy(entrypoint):
    arguments = ["--architecture", CORE, "--config", "unused", "--checkpoint", "unused",
                 "--output", "runs/unused", "--legacy-weights", "--released-weights"]
    if entrypoint == "infer.py":
        arguments += ["--input", "unused"]
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / entrypoint), *arguments],
                            text=True, capture_output=True)
    assert result.returncode == 2 and "not allowed with argument" in result.stderr


def test_release_manifest_matches_pinned_loader():
    manifest = json.loads((ROOT / "weights.json").read_text())
    assert manifest["architecture"] == CORE and manifest["fusion_weights_available"] is False
    assert manifest["datasets"] == ["arctic", "hot3d"]
    assert {asset["solver"] for asset in manifest["assets"]} == {"standard", "kfree"}
    for asset in manifest["assets"]:
        assert asset["sha256"] == release.RELEASE_SHA256[asset["solver"]]
        assert asset["source_checkpoint_sha256"] == LEGACY_CORE_SHA256[asset["solver"]]
