from __future__ import annotations

from types import SimpleNamespace

import torch

from dreamhand.camera import project_pinhole
from dreamhand.evaluator import EvaluationAccumulator, score_batch


def test_perfect_detection_and_pose_scores_zero_error() -> None:
    batch_size, frames = 1, 4
    generator = torch.Generator().manual_seed(7)
    joints_root = torch.randn(batch_size, 1, 2, 21, 3, generator=generator) * 0.02
    joints_root = joints_root.expand(batch_size, frames, 2, 21, 3).clone()
    joints_root[..., 0, :] = 0.0
    translation = torch.tensor([0.02, -0.01, 1.0]).view(1, 1, 1, 3).expand(1, frames, 2, 3)
    joints_camera = joints_root + translation.unsqueeze(-2)
    intrinsics = torch.tensor([[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]])
    image_size = torch.tensor([[480.0, 640.0]])
    joints_2d = project_pinhole(joints_camera, intrinsics, image_size)
    valid_hand = torch.zeros(batch_size, frames, 2, dtype=torch.bool)
    valid_hand[..., 0] = True
    valid_joints = valid_hand.unsqueeze(-1).expand(-1, -1, -1, 21)
    identity = torch.eye(3).expand(batch_size, frames, 2, 3, 3).clone()
    vertices_root = joints_root.index_select(-2, torch.arange(778) % joints_root.shape[-2])
    decoder = SimpleNamespace(
        existence_logits=torch.tensor([10.0, -10.0]).view(1, 1, 2).expand(1, frames, 2),
        anchors_2d=joints_2d,
        global_rotation=identity,
    )
    output = SimpleNamespace(
        decoder=decoder,
        joints_camera=joints_camera,
        joints_root_mano=joints_root,
        vertices_camera=vertices_root + translation.unsqueeze(-2),
        pnp=SimpleNamespace(translation=translation),
        mano_translation=translation,
    )
    batch = {
        "dataset": "arctic",
        "intrinsics": intrinsics,
        "image_size": image_size,
        "distortion": torch.zeros(1, 8),
        "camera_model": "pinhole",
        "camera_parameters": None,
        "source_image_size": None,
        "joints_camera": joints_camera,
        "joints_root": joints_root,
        "translation": translation,
        "joints_2d": joints_2d,
        "global_rotation": identity,
        "existence": valid_hand,
        "valid_hand": valid_hand,
        "valid_mano": valid_hand,
        "valid_joints_3d": valid_joints,
        "valid_joints_2d": valid_joints,
    }
    accumulator = EvaluationAccumulator()
    score_batch(
        accumulator,
        output,
        batch,
        vertices_root,
        joints_root[0, 0],
        solver="standard",
        gt_mano_translation=translation,
    )
    result = accumulator.finalize()
    assert result["true_positive"] == frames
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["FAcc"] == 1.0
    assert result["MPJPE-p_mm"] == 0.0
    assert result["PA-p_mm"] < 1e-4
    assert result["EPE2D-p_px"] == 0.0
    assert result["GO-p_deg"] == 0.0
    assert result["CT-p_m"] == 0.0
    assert result["Wrist-p_m"] == 0.0
    assert result["Jitter_mm_per_frame2"] < 1e-4

    # A root-offset difference must affect native MANO CT, not wrist error.
    output.mano_translation = translation + translation.new_tensor([0.1, 0.0, 0.0])
    translated = EvaluationAccumulator()
    score_batch(
        translated,
        output,
        batch,
        vertices_root,
        joints_root[0, 0],
        solver="standard",
        gt_mano_translation=translation,
    )
    assert abs(translated.finalize()["CT-p_m"] - 0.1) < 1e-6
    assert translated.finalize()["Wrist-p_m"] == 0
