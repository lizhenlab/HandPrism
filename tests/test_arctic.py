from __future__ import annotations

import json

import torch

from handprism.data.arctic import _declared_image_offset, distort_camera_points


def test_zero_distortion_is_identity() -> None:
    points = torch.tensor([[0.1, -0.2, 1.0], [0.2, 0.3, 2.0]])
    torch.testing.assert_close(distort_camera_points(points, torch.zeros(8)), points)


def test_radial_distortion_preserves_depth_and_sign() -> None:
    points = torch.tensor([[0.3, -0.2, 1.0]])
    coefficients = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    distorted = distort_camera_points(points, coefficients)
    assert distorted[0, 0] > points[0, 0]
    assert distorted[0, 1] < points[0, 1]
    torch.testing.assert_close(distorted[..., 2], points[..., 2])


def test_image_offset_comes_from_release_metadata(tmp_path) -> None:
    path = tmp_path / "data" / "meta"
    path.mkdir(parents=True)
    (path / "misc.json").write_text(json.dumps({"s02": {"ioi_offset": 2}}))
    assert _declared_image_offset(tmp_path, "s02") == 2
