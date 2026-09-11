"""ARCTIC view-0 adapter using the release's MANO and distorted egocamera labels."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from ..mano import SmplxMano
from ..rotations import axis_angle_to_matrix
from .contract import HandPrismSample, validate_sample
from .policy import allowed_path


RAW_HEIGHT = 2000
RAW_WIDTH = 2800
TARGET_HEIGHT = 480
TARGET_WIDTH = 672


def distort_camera_points(points: Tensor, coefficients: Tensor) -> Tensor:
    """ARCTIC/OpenCV rational+tangential eight-parameter distortion."""

    if coefficients.numel() != 8:
        raise ValueError("ARCTIC dist8 must contain eight values")
    original_dtype = points.dtype
    points = points.double()
    k = coefficients.double().flatten()
    z = points[..., 2]
    safe_z = torch.where(z.abs() > 1e-9, z, torch.full_like(z, 1e-9))
    x, y = points[..., 0] / safe_z, points[..., 1] / safe_z
    x2, y2, xy = x.square(), y.square(), x * y
    r2 = x2 + y2
    r4, r6 = r2.square(), r2.square() * r2
    radial = (1 + k[0] * r2 + k[1] * r4 + k[4] * r6) / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6)
    xd = x * radial + 2 * k[2] * xy + k[3] * (r2 + 2 * x2)
    yd = y * radial + 2 * k[3] * xy + k[2] * (r2 + 2 * y2)
    return torch.stack((xd * z, yd * z, z), dim=-1).to(original_dtype)


def _project(points: Tensor, intrinsics: Tensor) -> Tensor:
    z = points[..., 2].clamp_min(1e-8)
    u = intrinsics[0, 0] * points[..., 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points[..., 1] / z + intrinsics[1, 2]
    return torch.stack((u / TARGET_WIDTH, v / TARGET_HEIGHT), dim=-1)


def _load_dict(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


@lru_cache(maxsize=8)
def _cached_mano(model_path: str, flat_hand_mean: bool) -> SmplxMano:
    return SmplxMano(model_path, flat_hand_mean=flat_hand_mean)


def _declared_image_offset(root: Path, subject: str) -> int:
    metadata = json.loads((root / "data" / "meta" / "misc.json").read_text())
    try:
        offset = int(metadata[subject]["ioi_offset"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ARCTIC ioi_offset is missing for {subject}") from error
    if offset < 0:
        raise ValueError(f"ARCTIC ioi_offset must be non-negative, got {offset}")
    return offset


def _decode_images(image_root: Path, indices: Tensor, image_offset: int,
                   detail_long_side: int = 0) -> Tensor | tuple[Tensor, Tensor]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise ImportError("ARCTIC image loading requires Pillow") from error
    if not image_root.is_dir():
        raise FileNotFoundError(f"no RGB frames under {image_root}")
    frames = []
    detail = []
    for index in indices.tolist():
        stem = f"{index + image_offset:05d}"
        path = next(
            (
                image_root / f"{stem}{suffix}"
                for suffix in (".jpg", ".jpeg", ".png")
                if (image_root / f"{stem}{suffix}").is_file()
            ),
            None,
        )
        if path is None:
            raise FileNotFoundError(f"missing ARCTIC frame {stem} under {image_root}")
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if detail_long_side:
            scale = min(1., detail_long_side / max(image.size))
            native = image.resize(tuple(round(size * scale) for size in image.size), Image.Resampling.BILINEAR)
            detail.append(torch.from_numpy(np.asarray(native).copy()).permute(2, 0, 1))
        image = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BILINEAR)
        frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
    video = torch.stack(frames, dim=1).float().div(127.5).sub(1.0)
    return (video, torch.stack(detail, dim=1)) if detail_long_side else video


def _decode_raw_parameters(
    annotation: dict[str, Any],
    indices: Tensor,
    raw_mano: SmplxMano,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    global_world, articulation, betas, roots, valid = [], [], [], [], []
    for side, layer in (("left", raw_mano.left), ("right", raw_mano.right)):
        record = annotation[side]
        count = indices.numel()
        shape = torch.as_tensor(record["shape"], dtype=torch.float32)
        shape = shape[None].expand(count, -1) if shape.ndim == 1 else shape.index_select(0, indices)
        result = layer(
            global_orient=torch.as_tensor(record["rot"], dtype=torch.float32).index_select(
                0, indices
            ),
            hand_pose=torch.as_tensor(record["pose"], dtype=torch.float32).index_select(0, indices),
            betas=shape,
            transl=torch.as_tensor(record["trans"], dtype=torch.float32).index_select(0, indices),
            return_verts=True,
            return_full_pose=True,
        )
        full_pose = result.full_pose.float()
        global_world.append(axis_angle_to_matrix(full_pose[:, :3]))
        articulation.append(axis_angle_to_matrix(full_pose[:, 3:].reshape(count, 15, 3)))
        betas.append(shape)
        roots.append(result.joints[:, 0].float())
        fitting = torch.as_tensor(record["fitting_err"], dtype=torch.float32).index_select(
            0, indices
        )
        valid.append(torch.isfinite(result.vertices).all((-1, -2)) & torch.isfinite(fitting))
    return (
        torch.stack(global_world, dim=1),
        torch.stack(articulation, dim=1),
        torch.stack(betas, dim=1),
        torch.stack(roots, dim=1),
        torch.stack(valid, dim=1),
    )


def load_arctic_window(
    root: str | Path,
    sequence: str,
    start: int,
    frames: int,
    *,
    mano_model_path: str | Path,
    image_offset: int | None = None,
    detail_long_side: int = 0,
    extended_contract: bool = False,
    decode_rgb: bool = True,
) -> HandPrismSample:
    """Load a contiguous release-indexed window; no temporal subsampling."""

    root = allowed_path(root)
    subject, name = sequence.split("/", 1)
    declared_offset = _declared_image_offset(root, subject)
    if image_offset is None:
        image_offset = declared_offset
    elif int(image_offset) != declared_offset:
        raise ValueError(
            f"ARCTIC manifest ioi_offset {image_offset} disagrees with "
            f"{subject} metadata value {declared_offset}"
        )
    indices = torch.arange(start, start + frames, dtype=torch.long)
    mano_annotation = _load_dict(root / "data" / "raw_seqs" / subject / f"{name}.mano.npy")
    ego = _load_dict(root / "data" / "raw_seqs" / subject / f"{name}.egocam.dist.npy")
    if int(indices[-1]) >= len(mano_annotation["left"]["rot"]):
        raise IndexError("ARCTIC window exceeds annotation length")

    resolved_mano = str(Path(mano_model_path).resolve())
    raw_mano = _cached_mano(resolved_mano, False)
    canonical_mano = _cached_mano(resolved_mano, True)
    global_world, articulation, betas, root_world, valid = _decode_raw_parameters(
        mano_annotation, indices, raw_mano
    )
    rotation = torch.as_tensor(ego["R_k_cam_np"], dtype=torch.float32).index_select(0, indices)
    camera_shift = (
        torch.as_tensor(ego["T_k_cam_np"], dtype=torch.float32)
        .index_select(0, indices)
        .reshape(-1, 3)
    )
    global_camera = torch.einsum("tij,tsjk->tsik", rotation, global_world)
    root_camera = torch.einsum("tij,tsj->tsi", rotation, root_world) + camera_shift[:, None]
    with torch.no_grad():
        joints_root, _ = canonical_mano(global_camera, articulation, betas)
    joints_camera = joints_root + root_camera[..., None, :]

    intrinsics = torch.as_tensor(ego["intrinsics"], dtype=torch.float32).clone()
    intrinsics[0] *= TARGET_WIDTH / RAW_WIDTH
    intrinsics[1] *= TARGET_HEIGHT / RAW_HEIGHT
    if extended_contract:
        # PIL's resize maps pixel centers, not image corners. Core retains its
        # established calibration convention; Fusion's extended contract is
        # consistent with native-detail ROI sampling and the HOT3D adapter.
        intrinsics[0, 2] += .5 * TARGET_WIDTH / RAW_WIDTH - .5
        intrinsics[1, 2] += .5 * TARGET_HEIGHT / RAW_HEIGHT - .5
    distorted = distort_camera_points(joints_camera, torch.as_tensor(ego["dist8"]))
    joints_2d = _project(distorted, intrinsics)
    joint_valid = valid[..., None].expand(-1, -1, 21)
    visible = (
        joint_valid & (joints_camera[..., 2] > 0.01) & ((joints_2d >= 0) & (joints_2d < 1)).all(-1)
    )
    image_args = (root / "data" / "images" / sequence / "0", indices, int(image_offset))
    rgb_high = None
    if not decode_rgb:
        video = torch.zeros(3, frames, 1, 1)  # Geometry-index tool only.
    elif detail_long_side:
        video, rgb_high = _decode_images(*image_args, detail_long_side)
    else:
        video = _decode_images(*image_args)
    sample = HandPrismSample(
        dataset="arctic",
        recording_id=sequence,
        frame_indices=indices,
        video=video,
        rgb_high=rgb_high,
        # Release frames are 30 Hz; ioi_offset affects image/label alignment,
        # not the annotation clock. This is a declared index clock, not PTS.
        timestamps=indices.double() / 30. if extended_contract else None,
        timestamp_source="arctic_release_index_30hz" if extended_contract else None,
        in_frame=visible if extended_contract else None,
        observed=torch.zeros_like(visible) if extended_contract else None,
        observed_valid=torch.zeros_like(visible) if extended_contract else None,
        # 'visibility' is geometric in-frame presence, not true occlusion.
        visibility_valid=torch.ones_like(valid) if extended_contract else None,
        intrinsics=intrinsics,
        image_size=torch.tensor([float(TARGET_HEIGHT), float(TARGET_WIDTH)]),
        global_rotation=global_camera,
        articulation=articulation,
        betas=betas,
        translation=root_camera,
        joints_root=joints_root,
        joints_camera=joints_camera,
        joints_2d=joints_2d,
        existence=valid,
        visibility=visible.any(-1),
        valid_hand=valid,
        valid_mano=valid,
        valid_joints_3d=joint_valid,
        valid_joints_2d=visible,
        valid_ray=torch.tensor(True),
        distortion=torch.as_tensor(ego["dist8"], dtype=torch.float32),
    )
    validate_sample(sample)
    return sample
