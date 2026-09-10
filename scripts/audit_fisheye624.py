#!/usr/bin/env python3
"""Compare the differentiable HOT3D projection against Project Aria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dreamhand.camera import project_fisheye624_upright
from dreamhand.data.hot3d import IMAGE_SIZE, RGB_STREAM, UPRIGHT_FROM_SOURCE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording",
        type=Path,
        default=Path("/data0/dataset/hot3d/P0001_10a27bf7"),
    )
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId

    provider = data_provider.create_vrs_data_provider(str(args.recording / "recording.vrs"))
    stream = StreamId(RGB_STREAM)
    label = provider.get_label_from_stream_id(stream)
    calibration = provider.get_device_calibration().get_camera_calib(label)
    source_width, source_height = calibration.get_image_size()
    parameters = torch.from_numpy(
        np.asarray(calibration.get_projection_params(), dtype=np.float32)
    ).unsqueeze(0)
    generator = torch.Generator().manual_seed(260820308)
    source_points = torch.randn(args.samples, 3, generator=generator)
    source_points[:, :2] *= 0.4
    source_points[:, 2] = torch.rand(args.samples, generator=generator) + 0.5
    upright_points = torch.einsum("ij,nj->ni", UPRIGHT_FROM_SOURCE, source_points)
    ours = project_fisheye624_upright(
        upright_points.reshape(1, args.samples, 1, 1, 3),
        parameters,
        torch.tensor([[float(source_height), float(source_width)]]),
        torch.tensor([[float(IMAGE_SIZE), float(IMAGE_SIZE)]]),
    ).reshape(args.samples, 2)
    official, retained = [], []
    for index, point in enumerate(source_points.numpy().astype(np.float64)):
        pixel = calibration.project_no_checks(point)
        if pixel is None:
            continue
        source_u, source_v = np.asarray(pixel)
        target_u = (source_height - 0.5 - source_v) * IMAGE_SIZE / source_height - 0.5
        target_v = (source_u + 0.5) * IMAGE_SIZE / source_width - 0.5
        official.append([target_u / IMAGE_SIZE, target_v / IMAGE_SIZE])
        retained.append(index)
    difference_px = (
        ours[retained] - torch.tensor(official, dtype=ours.dtype)
    ).norm(dim=-1) * IMAGE_SIZE
    report = {
        "recording": args.recording.name,
        "camera_model": str(calibration.get_model_name()),
        "samples": len(retained),
        "mean_error_px": float(difference_px.mean()),
        "max_error_px": float(difference_px.max()),
        "status": "pass" if float(difference_px.max()) < 1e-3 else "fail",
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
