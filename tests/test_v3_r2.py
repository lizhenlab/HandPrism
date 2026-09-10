"""Pretraining fixes: no production images, no training jobs, no v2 writes."""

from pathlib import Path
from types import SimpleNamespace
import os

import pytest
import torch
import torch.nn.functional as F

from dreamhand.config import DecoderConfig, SolverConfig
from dreamhand.decoder import DreamHandDecoder
from dreamhand.losses import camera_fit_bearing_loss, camera_fit_supervision, masked_clip_mean
from dreamhand.mano import ToyMano, correct_left_shapedirs
from dreamhand.model import DreamHandModel
from dreamhand.paths import v3_run_path
from dreamhand.ray import fit_effective_pinhole_camera, normalized_pixel_grid
from scripts.train import decoder_config_from_json, load_config, restore_checkpoint


def tiny_config(**kwargs):
    return DecoderConfig(feature_dim=24, hidden_dim=16, layers=2, heads=4, ffn_dim=32, **kwargs)


@pytest.mark.parametrize("solver", ["standard", "kfree"])
def test_amp_geometry_is_fp32_finite_and_differentiable(solver):
    torch.manual_seed(43)
    model = DreamHandModel(ToyMano(), tiny_config(), architecture="handprism-fusion").eval()
    features = torch.randn(1, 3, 15, 21, 24, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(
            features,
            target_frames=7,
            solver=solver,
            image_size=torch.tensor([[480.0, 672.0]]),
            intrinsics=torch.tensor([[[450.0, 0, 336.0], [0, 450.0, 240.0], [0, 0, 1.0]]]),
        )
    values = [
        out.decoder.anchors_2d,
        out.decoder.global_rotation,
        out.decoder.articulation,
        out.decoder.joints_root_direct,
        out.decoder.attention_heatmaps,
        out.ray_field,
        out.joints_root_mano,
        out.vertices_camera,
        out.pnp.translation,
        out.mano_translation,
    ]
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in values)
    rotation = out.decoder.global_rotation
    torch.testing.assert_close(
        rotation.transpose(-1, -2) @ rotation,
        torch.eye(3).expand_as(rotation),
        atol=2e-5,
        rtol=2e-5,
    )
    (out.joints_camera.square().mean() + out.decoder.anchors_2d.square().mean()).backward()
    assert torch.isfinite(features.grad).all()
    assert model.decoder.pose_head[-1].weight.grad.abs().sum() > 0


def test_left_shapedirs_fix_is_shared_and_idempotent():
    left = SimpleNamespace(shapedirs=torch.ones(5, 3, 2))
    right = SimpleNamespace(shapedirs=torch.ones(5, 3, 2))
    assert correct_left_shapedirs(left, right)
    assert (left.shapedirs[:, 0] == -1).all()
    assert (left.shapedirs[:, 1:] == 1).all()
    assert not correct_left_shapedirs(left, right)
    assert (left.shapedirs[:, 0] == -1).all()
    assert (right.shapedirs == 1).all()


def test_left_shapedirs_already_corrected_asset_is_unchanged():
    left = SimpleNamespace(shapedirs=-torch.ones(5, 3, 2))
    right = SimpleNamespace(shapedirs=torch.ones(5, 3, 2))
    before = left.shapedirs.clone()
    assert not correct_left_shapedirs(left, right)
    torch.testing.assert_close(before, left.shapedirs)


def test_clip_reduction_matches_accumulation_and_equal_rank_partition():
    value = torch.tensor(
        [[1.0, 9.0, 9.0], [3.0, 3.0, 9.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
    )
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.bool)
    full = masked_clip_mean(value, mask)
    accumulated = sum(masked_clip_mean(value[i : i + 1], mask[i : i + 1]) for i in range(4)) / 4
    rank_mean = (masked_clip_mean(value[:2], mask[:2]) + masked_clip_mean(value[2:], mask[2:])) / 2
    torch.testing.assert_close(full, torch.tensor(2.25))
    for other in (accumulated, rank_mean):
        torch.testing.assert_close(full, other)
        torch.testing.assert_close(
            torch.autograd.grad(full, value, retain_graph=True)[0],
            torch.autograd.grad(other, value, retain_graph=True)[0],
        )


def test_clip_reduction_masks_invalid_nan_and_keeps_empty_clip_zero():
    value = torch.tensor([[float("nan"), 2.0], [float("nan"), float("nan")]], requires_grad=True)
    result = masked_clip_mean(value, torch.tensor([[False, True], [False, False]]))
    assert result == 1
    result.backward()
    assert torch.isfinite(value.grad).all()


def test_calibration_cache_is_bounded_keyed_by_calibration_and_returns_copies():
    import numpy as np
    from dreamhand.data.hot3d import _cached_ray_field, _RAY_FIELD_CACHE, RAY_GRID_SIZE

    class Calibration:
        def __init__(self, focal):
            self.focal = focal
            self.calls = 0

        def get_projection_params(self):
            return np.array([self.focal, *([0.0] * 14)])

        def unproject_no_checks(self, pixel):
            self.calls += 1
            return np.array([pixel[0] / self.focal, pixel[1] / self.focal, 1.0])

    _RAY_FIELD_CACHE.clear()
    calibration = Calibration(500.0)
    size = torch.tensor([1408.0, 1408.0])
    first = _cached_ray_field(calibration, size)
    baseline = first.clone()
    first.zero_()
    torch.testing.assert_close(_cached_ray_field(calibration, size), baseline)
    assert calibration.calls == RAY_GRID_SIZE**2
    other = _cached_ray_field(Calibration(600.0), size)
    assert not torch.allclose(baseline, other)
    for focal in range(700, 733):
        _cached_ray_field(Calibration(float(focal)), size)
    assert len(_RAY_FIELD_CACHE) == 32
    _RAY_FIELD_CACHE.clear()


def test_optional_anchor_offset_is_zero_initialized_and_cell_bounded():
    torch.manual_seed(7)
    base = DreamHandDecoder(tiny_config(), architecture="handprism-fusion").eval()
    refined = DreamHandDecoder(tiny_config(anchor_offset_cells=1), architecture="handprism-fusion").eval()
    missing, unexpected = refined.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"anchor_offset_head.weight", "anchor_offset_head.bias"}
    assert not unexpected
    queries = torch.randn(1, 2, 48, 16)
    weights = torch.zeros(1, 2, 48, 15 * 21)
    weights[..., -1] = 1
    original = base._readout(queries, weights, 5, 15, 21).anchors_2d
    initial = refined._readout(queries, weights, 5, 15, 21).anchors_2d
    torch.testing.assert_close(original, initial)
    with torch.no_grad():
        refined.anchor_offset_head.bias.fill_(1.0)
    shifted = refined._readout(queries, weights, 5, 15, 21).anchors_2d
    assert (shifted[..., 0] > 1 - 0.5 / 21).all()
    assert ((shifted - original).abs() <= torch.tensor([1 / 21, 1 / 15])).all()
    shifted.mean().backward()
    assert refined.anchor_offset_head.weight.grad.abs().sum() > 0


def fish_rays():
    grid = normalized_pixel_grid(15, 15, device=torch.device("cpu"), dtype=torch.float32)
    xy = grid - 0.5
    radius = xy.norm(dim=-1, keepdim=True)
    theta = radius / 0.6
    return torch.cat((xy / radius.clamp_min(1e-8) * theta.sin(), theta.cos()), -1)[None]


@pytest.mark.parametrize("target_mode", ["effective_camera", "raw_bearings"])
def test_opt_in_fit_targets_supervise_without_removing_inference_guard(target_mode):
    grid = normalized_pixel_grid(15, 15, device=torch.device("cpu"), dtype=torch.float32)
    predicted = F.normalize(torch.cat(((grid - 0.5) / 0.8, torch.ones(15, 15, 1)), -1), dim=-1)[
        None
    ].requires_grad_()
    target = fish_rays().requires_grad_()
    config = SolverConfig(camera_fit_target=target_mode)
    assert not fit_effective_pinhole_camera(target, config).valid.any()
    camera = fit_effective_pinhole_camera(predicted, config)
    assert camera_fit_bearing_loss(camera, target, None, "l1") == 0
    loss = camera_fit_bearing_loss(camera, target, None, "l1", config)
    assert loss > 0
    _, mask, activity = camera_fit_supervision(camera, target, torch.tensor([True]), config)
    assert mask.all() and activity["target_compatible_fraction"] == 0
    assert activity["supervised_ray_fraction"] == 1
    loss.backward()
    assert target.grad is None
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0


def test_v3_outputs_cannot_overwrite_v2_or_escape_through_symlinks(tmp_path):
    root = tmp_path / "v3"
    (root / "runs").mkdir(parents=True)
    assert v3_run_path(root, Path("runs/new_v3")) == root / "runs/new_v3"
    for path in [
        Path("runs/two_dataset_standard_v2_clean"),
        tmp_path / "v2",
        Path("runs/../README.md"),
    ]:
        with pytest.raises(ValueError):
            v3_run_path(root, path)
    (root / "runs/escape").symlink_to(tmp_path / "v2")
    with pytest.raises(ValueError):
        v3_run_path(root, Path("runs/escape/output"))


def test_old_v3_checkpoint_is_rejected_before_loading_parameters(tmp_path):
    path = tmp_path / "old-v3.pt"
    torch.save(
        {
            "format": "ace-ego-hand-two-dataset-checkpoint-v3",
            "implementation_id": "ace-ego-hand-independent-v3",
        },
        path,
    )
    from scripts.train import load_config
    config = load_config(Path(__file__).resolve().parents[1] / "configs/handprism_fusion_standard.json")
    with pytest.raises(ValueError, match="checkpoint architecture/implementation mismatch"):
        restore_checkpoint(path, None, None, None, 0, 1, config)


def test_both_configs_select_conservative_ablation_defaults():
    root = Path(__file__).parents[1]
    for name in ("two_dataset_v3.json", "two_dataset_kfree_v3.json"):
        config = load_config(root / "configs" / name)
        assert config["loss_reduction"] == "per_clip"
        assert config["geometry_dtype"] == "float32"
        assert decoder_config_from_json(config).anchor_offset_cells == 0
        assert set(config["dataset_roots"]) == {"arctic", "hot3d"}


def test_real_mano_pca_and_canonical_geometry_agree():
    asset = os.environ.get("MANO_MODEL_PATH")
    if not asset:
        pytest.skip("licensed MANO assets not configured")
    from dreamhand.data.hot3d import _mano_layers, _geometry
    from dreamhand.rotations import axis_angle_to_matrix

    canonical, left, right = _mano_layers(asset)
    torch.testing.assert_close(left.shapedirs, canonical.left.shapedirs)
    torch.testing.assert_close(right.shapedirs, canonical.right.shapedirs)
    generator = torch.Generator().manual_seed(20260910)
    shape = torch.randn(2, 10, generator=generator)
    annotations = [
        {
            str(slot): {
                "wrist_xform": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "t_xyz": [0.1, 0.2, 0.8]},
                "pose": torch.randn(15, generator=generator).tolist(),
            }
            for slot in range(2)
        }
    ]
    rotation = axis_angle_to_matrix(torch.tensor([[0.1, -0.2, 0.3]]))
    shift = torch.tensor([[0.02, -0.03, 0.04]])
    geometry = _geometry(annotations, rotation, shift, asset, shape)
    roots = canonical.root_offset(shape[None])
    _, canonical_vertices = canonical(
        geometry["global_rotation"], geometry["articulation"], geometry["betas"]
    )
    for slot, layer in enumerate((left, right)):
        reference = layer(
            betas=shape[slot : slot + 1],
            global_orient=torch.zeros(1, 3),
            hand_pose=torch.tensor(annotations[0][str(slot)]["pose"])[None],
            transl=torch.tensor([[0.1, 0.2, 0.8]]),
            return_verts=True,
        )
        expected = torch.einsum("tij,tvj->tvi", rotation, reference.vertices) + shift[:, None]
        actual = canonical_vertices[:, slot] + geometry["translation"][:, slot, None]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        # Independently check the analytic J0 convention against SMPL-X output.
        full = (canonical.left, canonical.right)[slot](
            betas=shape[slot : slot + 1],
            global_orient=torch.tensor([[0.2, 0.1, -0.3]]),
            hand_pose=torch.randn(1, 45, generator=generator) * 0.1,
            return_verts=True,
        )
        torch.testing.assert_close(roots[:, slot], full.joints[:, 0], atol=1e-6, rtol=1e-5)
