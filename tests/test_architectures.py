"""Architecture selection and preserved-weight compatibility; CPU only."""

import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import types

import numpy as np
import pytest
import torch

from dreamhand.architectures import (
    ARCHITECTURES, CORE, FUSION, CHECKPOINT_FORMAT, LEGACY_CORE_SHA256,
    add_architecture_argument, architecture_spec, validate_checkpoint_identity,
)
from dreamhand.attention import AlternatingLayer
from dreamhand.config import DecoderConfig, SolverConfig
from dreamhand.decoder import DreamHandDecoder
from dreamhand.mano import ToyMano
from dreamhand.model import DreamHandModel
from dreamhand.system import DreamHandSystem
from dreamhand.ray import fit_effective_pinhole_camera, normalized_pixel_grid
from dreamhand.losses import camera_fit_bearing_loss
from scripts.infer import load_clip, predict_clip
from scripts.evaluate import save_prediction
from scripts.run_pipeline import Supervisor
from scripts.train import (
    load_config, load_trainable_state, preflight_resume, restore_checkpoint, save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def config_for(architecture, solver="standard"):
    return load_config(
        ROOT / "configs" / f"{architecture.replace('-', '_')}_{solver}.json",
        architecture=architecture,
    )


def tiny_config():
    return DecoderConfig(feature_dim=24, hidden_dim=16, layers=2, heads=4, ffn_dim=32)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("solver", ["standard", "kfree"])
def test_named_config_routes_real_architecture(architecture, solver):
    config = config_for(architecture, solver)
    assert set(config["dataset_weights"]) == {"arctic", "hot3d"}
    assert config["architecture_contract"] == architecture_spec(architecture).contract
    model = DreamHandModel(ToyMano(), tiny_config(), architecture=architecture)
    assert model.decoder.layers[0].joint_time_query == (architecture == FUSION)
    features = torch.randn(1, 3, 4, 4, 24, requires_grad=True)
    k = torch.tensor([[[80., 0, 64], [0, 80, 64], [0, 0, 1]]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(features, target_frames=5, solver=solver, intrinsics=k,
                       image_size=torch.tensor([[128., 128.]]))
    assert output.decoder.anchors_2d.dtype == torch.float32
    assert output.pnp.residual_kind == (
        "bearing_diagonal_proxy" if architecture == CORE else "native_pixels"
    )
    loss = output.joints_camera.square().mean() + output.decoder.joints_root_direct.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_explicit_selection_is_mandatory():
    parser = argparse.ArgumentParser()
    add_architecture_argument(parser)
    for arguments in ([], ["--architecture", "v1"], ["--architecture", "auto"]):
        with pytest.raises(SystemExit):
            parser.parse_args(arguments)
    with pytest.raises(TypeError):
        DreamHandModel(ToyMano())
    with pytest.raises(TypeError):
        DreamHandDecoder()


@pytest.mark.parametrize("script", [
    "train.py", "evaluate.py", "run_pipeline.py", "check_readiness.py", "infer.py",
    "train_three_dataset.py", "evaluate_three_dataset.py", "run_full_reproduction.py",
    "audit_readiness.py", "audit_three_dataset_readiness.py",
])
def test_production_cli_refuses_omitted_architecture(script):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script)],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert "--architecture" in result.stderr and "required" in result.stderr


def test_core_and_fusion_have_different_query_connectivity():
    torch.manual_seed(1)
    core = AlternatingLayer(16, 4, 32, joint_time_query=False).eval()
    fusion = AlternatingLayer(16, 4, 32, joint_time_query=True).eval()
    fusion.load_state_dict(core.state_dict())
    q = torch.randn(1, 3, 4, 16)
    memory = torch.randn(1, 3, 5, 16)
    changed = q.clone()
    changed[:, 0, 0, 0] += 4
    assert torch.equal(core(q, memory)[0][:, :, 1:], core(changed, memory)[0][:, :, 1:])
    assert not torch.allclose(fusion(q, memory)[0][:, :, 1:], fusion(changed, memory)[0][:, :, 1:])


def archived_modules(archive, monkeypatch):
    # Load only these code members in an isolated in-memory namespace. Never
    # extract archive paths, modify the archive, or import dataset adapters.
    package = "_handprism_core_reference"
    parent = types.ModuleType(package)
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, package, parent)
    modules = {}
    with tarfile.open(archive, "r:gz") as source:
        for name in ("config", "positional", "rotations", "attention", "decoder", "ray", "model"):
            member = f"src/dreamhand/{name}.py"
            module = types.ModuleType(f"{package}.{name}")
            module.__package__ = package
            monkeypatch.setitem(sys.modules, module.__name__, module)
            exec(compile(source.extractfile(member).read(), member, "exec"), module.__dict__)
            modules[name] = module
    return modules


@pytest.mark.parametrize("solver,archive", [
    ("standard", "standard_db39b89_source.tar.gz"),
    ("kfree", "kfree_c2823d4_source.tar.gz"),
])
def test_core_fp32_matches_actual_preserved_architecture(solver, archive, monkeypatch):
    source = ROOT / "preserved_v2" / archive
    if not source.exists():
        source = ROOT / "preserved_v2_20260910" / archive
    if not source.exists():
        pytest.skip("optional historical source fixture is not distributed")
    modules = archived_modules(source, monkeypatch)
    old_config = modules["config"].DecoderConfig(feature_dim=24, hidden_dim=16, layers=2, heads=4, ffn_dim=32)
    torch.manual_seed(93)
    old = modules["model"].DreamHandModel(ToyMano(), old_config).eval()
    # Exercise nonconstant ray fields rather than only the zero-init fallback.
    torch.nn.init.normal_(old.ray_head.projection.weight, std=0.01)
    core = DreamHandModel(ToyMano(), tiny_config(), architecture=CORE).eval()
    core.load_state_dict(old.state_dict(), strict=True)
    features = torch.randn(1, 3, 4, 4, 24)
    kwargs = dict(target_frames=5, solver=solver,
                  intrinsics=torch.tensor([[[80., 0, 64], [0, 80, 64], [0, 0, 1]]]),
                  image_size=torch.tensor([[128., 128.]]))
    with torch.no_grad():
        reference, output = old(features, **kwargs), core(features, **kwargs)
    for field in ("anchors_2d", "joints_root_direct", "global_rotation", "articulation", "betas", "log_depth"):
        torch.testing.assert_close(getattr(output.decoder, field), getattr(reference.decoder, field), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output.joints_camera, reference.joints_camera)
    torch.testing.assert_close(output.pnp.translation, reference.pnp.translation)
    assert torch.equal(output.pnp.solved, reference.pnp.solved)


def test_core_does_not_use_fusion_rms_gate():
    grid = normalized_pixel_grid(8, 8, device=torch.device("cpu"), dtype=torch.float32)
    xy = grid - 0.5
    xy = xy * (1 + 3 * xy.square().sum(-1, keepdim=True))
    ray = torch.nn.functional.normalize(torch.cat((xy, torch.ones_like(xy[..., :1])), -1)[None], dim=-1)
    core = fit_effective_pinhole_camera(ray, SolverConfig(architecture=CORE))
    fusion = fit_effective_pinhole_camera(ray, SolverConfig(architecture=FUSION))
    assert core.valid.item() and not fusion.valid.item()
    assert core.rms_normalized.item() > 0.01
    pred = ray.clone().requires_grad_()
    fit = fit_effective_pinhole_camera(pred, SolverConfig(architecture=CORE))
    loss = camera_fit_bearing_loss(fit, ray, None, "l1", SolverConfig(architecture=CORE, camera_fit_target="core_bearings"))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def checkpoint_for(config):
    return {"format": CHECKPOINT_FORMAT, "architecture": config["architecture"],
            "implementation_id": config["implementation_id"], "config": copy.deepcopy(config)}


def test_checkpoint_architecture_mismatch_is_not_resolved_by_shapes():
    core, fusion = config_for(CORE), config_for(FUSION)
    for field in ("architecture", "implementation_id", "format", "config"):
        checkpoint = checkpoint_for(core)
        checkpoint[field] = checkpoint_for(fusion)[field] if field != "format" else "old"
        with pytest.raises(ValueError):
            validate_checkpoint_identity(checkpoint, core, architecture=CORE)
    with pytest.raises(ValueError):
        validate_checkpoint_identity(checkpoint_for(core), fusion, architecture=CORE)


def test_legacy_weights_require_explicit_core_and_pinned_digest():
    config = config_for(CORE)
    old = {"config": config, "step": 20000}
    with pytest.raises(ValueError):
        validate_checkpoint_identity(old, config, architecture=CORE)
    with pytest.raises(ValueError):
        validate_checkpoint_identity(old, config, architecture=CORE, legacy_sha256="unverified")
    validate_checkpoint_identity(old, config, architecture=CORE, legacy_sha256=LEGACY_CORE_SHA256["standard"])
    with pytest.raises(ValueError):
        validate_checkpoint_identity(old, config_for(FUSION), architecture=FUSION, legacy_sha256=LEGACY_CORE_SHA256["standard"])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_checkpoint_roundtrip_and_cpu_resume(tmp_path, monkeypatch, architecture):
    config = config_for(architecture)
    module = torch.nn.Linear(2, 2)
    module.architecture = architecture
    optimizer = torch.optim.AdamW(module.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    weights = {k: v.detach().clone() for k, v in module.named_parameters()}
    rng = {"torch": torch.get_rng_state(), "cuda": [], "numpy": np.random.get_state(), "python": random.getstate()}
    path = save_checkpoint(tmp_path, 12, module, optimizer, scheduler, 1.0, 1, config, [rng])
    assert json.loads((path.parent / "latest.json").read_text())["architecture"] == architecture
    preflight_resume(path, config, 1)
    with torch.no_grad():
        module.weight.zero_()
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)
    step, best = restore_checkpoint(path, module, optimizer, scheduler, 0, 1, config)
    assert (step, best) == (12, 1.0)
    for key, value in module.named_parameters():
        assert torch.equal(value, weights[key])
    other = FUSION if architecture == CORE else CORE
    with pytest.raises(ValueError):
        restore_checkpoint(path, module, optimizer, scheduler, 0, 1, config_for(other))


def test_resume_architecture_is_checked_before_cuda_or_data(tmp_path, monkeypatch):
    from scripts import train as training

    path = tmp_path / "mismatched.pt"
    checkpoint = checkpoint_for(config_for(CORE))
    checkpoint["world_size"] = 1
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="world size"):
        preflight_resume(path, config_for(CORE), 8)

    def forbidden(*args, **kwargs):
        pytest.fail("mismatched checkpoint reached CUDA or dataset initialization")

    monkeypatch.setattr(training, "distributed_context", forbidden)
    monkeypatch.setattr(training, "make_loaders", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "train_three_dataset.py", "--architecture", FUSION,
        "--config", "configs/handprism_fusion_standard.json",
        "--run-dir", "runs/architecture_preflight_never_written", "--resume", str(path),
    ])
    with pytest.raises(ValueError, match="architecture/implementation mismatch"):
        training.main()


def test_trainable_loading_rejects_broadcastable_wrong_shapes():
    module = torch.nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match="tensor shapes"):
        load_trainable_state(module, {"weight": torch.ones(1, 3), "bias": torch.ones(2)})


def test_supervisor_propagates_architecture_and_protects_other_run(tmp_path):
    supervisor = Supervisor(tmp_path, tmp_path / "control", 8, CORE)
    command = supervisor.distributed("scripts/train.py")
    assert command[command.index("--architecture") + 1] == CORE
    supervisor.set_state("running")
    with pytest.raises(ValueError, match="different or unnamed"):
        Supervisor(tmp_path, tmp_path / "control", 8, FUSION)


def test_prepared_clip_requires_calibration_only_for_standard(tmp_path):
    path = tmp_path / "clip.npz"
    video = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    np.savez(path, video=video)
    free = load_clip(path, "kfree")
    assert free["intrinsics"] is None and free["video"].shape == (1, 3, 5, 64, 64)
    with pytest.raises(ValueError, match="requires intrinsics"):
        load_clip(path, "standard")
    np.savez(path, video=video, intrinsics=np.array([[80, 0, 32], [0, 80, 32], [0, 0, 1]]))
    assert load_clip(path, "standard")["intrinsics"].shape == (1, 3, 3)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("solver", ["standard", "kfree"])
def test_clip_inference_and_export_keep_selected_architecture(tmp_path, architecture, solver):
    class SmallEncoder(torch.nn.Module):
        def forward(self, latent):
            return latent.new_ones((latent.shape[0], 2, 2, 2, 24))

    clip_path = tmp_path / "synthetic.npz"
    np.savez(clip_path, video=np.zeros((5, 64, 64, 3), dtype=np.uint8),
             intrinsics=np.array([[80, 0, 32], [0, 80, 32], [0, 0, 1]]))
    config = config_for(architecture, solver)
    batch = load_clip(clip_path, solver)
    system = DreamHandSystem(SmallEncoder(), ToyMano(), tiny_config(), architecture=architecture).eval()
    output, batch = predict_clip(system, torch.nn.Identity(), batch, config, torch.device("cpu"))
    destination = tmp_path / "prediction.npz"
    save_prediction(destination, output, batch, 20000, solver, architecture)
    with np.load(destination, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        assert metadata["architecture"] == architecture
        assert metadata["implementation_id"] == config["implementation_id"]
        assert metadata["solver"] == solver and metadata["checkpoint_step"] == 20000
        assert metadata["pnp_residual_kind"] == output.pnp.residual_kind
        assert data["joints_camera"].shape == (5, 2, 21, 3)
        assert np.isfinite(data["joints_camera"]).all()
        assert data["joints_camera"].dtype == np.float32


def test_wrong_named_config_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    config = config_for(CORE)
    for field, value in (("architecture", FUSION), ("architecture_contract", architecture_spec(FUSION).contract)):
        wrong = copy.deepcopy(config)
        wrong[field] = value
        path.write_text(json.dumps(wrong))
        with pytest.raises(ValueError):
            load_config(path, architecture=CORE)
