"""HOT3D Aria VRS adapter with native fisheye images and released MANO.

The native Fisheye624 view supplies calibrated ray supervision. Frozen
manifests split eligible labelled Aria recordings by participant and require
contiguous valid-mask intervals. Recordings without released pose ground
truth are not training or evaluation labels.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict
import csv
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from ..camera import FISHEYE624_UPRIGHT, project_fisheye624_upright
from ..mano import SmplxMano, correct_left_shapedirs, matrix_to_axis_angle
from ..rotations import axis_angle_to_matrix
from .contract import DreamHandSample, validate_sample
from .policy import allowed_path


RGB_STREAM = "214-1"
IMAGE_SIZE = 480
REQUIRED_TRAINING_MASKS = (
    "mask_hand_pose_available",
    "mask_headset_pose_available",
    "mask_good_exposure",
    "mask_qa_pass",
)
# Official Wan VAE (16x) followed by the released patch embedding (2x).
RAY_GRID_SIZE = IMAGE_SIZE // 32
UPRIGHT_FROM_SOURCE = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_RAY_FIELD_CACHE: OrderedDict[tuple, Tensor] = OrderedDict()


def quaternion_wxyz_to_matrix(quaternion: Tensor) -> Tensor:
    quaternion = F.normalize(quaternion.float(), dim=-1)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def upright_camera_transform(world_from_camera: Tensor) -> tuple[Tensor, Tensor]:
    camera_from_world = world_from_camera[:3, :3].transpose(0, 1)
    translation = -camera_from_world @ world_from_camera[:3, 3]
    clockwise = UPRIGHT_FROM_SOURCE.to(world_from_camera)
    return clockwise @ camera_from_world, clockwise @ translation


def _upright_intrinsics(parameters: Tensor, source_size: Tensor) -> Tensor:
    source_height, source_width = source_size
    scale_x = IMAGE_SIZE / source_height
    scale_y = IMAGE_SIZE / source_width
    focal, source_cx, source_cy = parameters[:3]
    target_cx = (source_height - 0.5 - source_cy) * scale_x - 0.5
    target_cy = (source_cx + 0.5) * scale_y - 0.5
    return torch.tensor(
        [
            [focal * scale_x, 0.0, target_cx],
            [0.0, focal * scale_y, target_cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _calibrated_ray_field(calibration: Any, source_size: Tensor) -> Tensor:
    """Unproject native fisheye pixels at the exact DiT token centers."""

    source_height, source_width = (float(value) for value in source_size.tolist())
    rows = []
    for row in range(RAY_GRID_SIZE):
        rays = []
        target_v = (row + 0.5) * IMAGE_SIZE / RAY_GRID_SIZE
        source_u = (target_v + 0.5) * source_width / IMAGE_SIZE - 0.5
        for column in range(RAY_GRID_SIZE):
            target_u = (column + 0.5) * IMAGE_SIZE / RAY_GRID_SIZE
            source_v = source_height - 0.5 - (target_u + 0.5) * source_height / IMAGE_SIZE
            ray = calibration.unproject_no_checks(
                np.asarray([source_u, source_v], dtype=np.float64)
            )
            if ray is None:
                raise RuntimeError("Fisheye624 failed to unproject an in-image token")
            upright = UPRIGHT_FROM_SOURCE @ torch.from_numpy(np.asarray(ray)).float()
            rays.append(F.normalize(upright, dim=0))
        rows.append(torch.stack(rays))
    return torch.stack(rows)


def _resize_native_image(image: np.ndarray) -> Tensor:
    upright = np.ascontiguousarray(np.rot90(image, k=3))
    tensor = torch.from_numpy(upright).permute(2, 0, 1).float()
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def _cached_ray_field(calibration: Any, source_size: Tensor) -> Tensor:
    """Bounded CPU-only calibration cache; no VRS handles cross workers."""

    key = (
        tuple(np.asarray(calibration.get_projection_params(), dtype=np.float64).tolist()),
        tuple(source_size.tolist()),
        RAY_GRID_SIZE,
    )
    if key not in _RAY_FIELD_CACHE:
        _RAY_FIELD_CACHE[key] = _calibrated_ray_field(calibration, source_size)
        if len(_RAY_FIELD_CACHE) > 32:
            _RAY_FIELD_CACHE.popitem(last=False)
    _RAY_FIELD_CACHE.move_to_end(key)
    return _RAY_FIELD_CACHE[key].clone()


def _closest(mapping: dict[int, Any], ordered: tuple[int, ...], timestamp: int) -> Any:
    index = bisect_left(ordered, timestamp)
    candidates = ordered[max(0, index - 1) : min(len(ordered), index + 1)]
    if not candidates:
        raise ValueError("HOT3D timestamped annotation is empty")
    selected = min(candidates, key=lambda value: abs(value - timestamp))
    return mapping[selected]


def _mask_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid HOT3D mask value {value!r}")
    return normalized == "true"


@lru_cache(maxsize=128)
def stream_mask(mask_path: str, stream_id: str = RGB_STREAM) -> dict[int, bool]:
    with Path(mask_path).open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row["timestamp[ns]"]): _mask_value(row["mask"])
            for row in csv.DictReader(handle)
            if row["stream_id"] == stream_id and int(row["timestamp[ns]"]) > 0
        }


@lru_cache(maxsize=32)
def stream_timestamps(mask_path: str, stream_id: str = RGB_STREAM) -> tuple[int, ...]:
    return tuple(stream_mask(mask_path, stream_id))


@lru_cache(maxsize=8)
def _parse_headset(path: str) -> tuple[dict[int, Tensor], tuple[int, ...]]:
    transforms: dict[int, Tensor] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            transform = torch.eye(4)
            transform[:3, :3] = quaternion_wxyz_to_matrix(
                torch.tensor(
                    [
                        float(row["q_wo_w"]),
                        float(row["q_wo_x"]),
                        float(row["q_wo_y"]),
                        float(row["q_wo_z"]),
                    ]
                )
            )
            transform[:3, 3] = torch.tensor(
                [
                    float(row["t_wo_x[m]"]),
                    float(row["t_wo_y[m]"]),
                    float(row["t_wo_z[m]"]),
                ]
            )
            transforms[int(row["timestamp[ns]"])] = transform
    return transforms, tuple(sorted(transforms))


@lru_cache(maxsize=8)
def _parse_mano(path: str) -> tuple[dict[int, dict[str, Any]], Tensor]:
    poses: dict[int, dict[str, Any]] = {}
    betas: list[Tensor | None] = [None, None]
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            poses[int(item["timestamp_ns"])] = item["hand_poses"]
            for slot, side in enumerate(("0", "1")):
                hand = item["hand_poses"].get(side)
                if betas[slot] is None and hand and "betas" in hand:
                    betas[slot] = torch.tensor(hand["betas"], dtype=torch.float32)
    complete = [torch.zeros(10) if value is None else value for value in betas]
    return poses, torch.stack(complete)


def _mano_root(model_path: str) -> Path:
    path = Path(model_path)
    if path.name.lower() == "mano":
        return path.parent
    if (path / "MANO_LEFT.pkl").is_file():
        return path.parent
    return path


@lru_cache(maxsize=4)
def _mano_layers(model_path: str):
    canonical = SmplxMano(model_path, flat_hand_mean=True)
    import smplx

    common = dict(
        model_path=str(_mano_root(model_path)),
        model_type="mano",
        use_pca=True,
        num_pca_comps=15,
        flat_hand_mean=False,
        num_betas=10,
        create_global_orient=False,
        create_hand_pose=False,
        create_betas=False,
        create_transl=False,
    )
    left = smplx.create(is_rhand=False, **common)
    right = smplx.create(is_rhand=True, **common)
    correct_left_shapedirs(left, right)
    return canonical, left, right


def _geometry(
    annotations: list[dict[str, Any]],
    rotation_camera_from_world: Tensor,
    translation_camera_from_world: Tensor,
    model_path: str,
    shape: Tensor,
) -> dict[str, Tensor]:
    frames = len(annotations)
    canonical, pca_left, pca_right = _mano_layers(str(Path(model_path).resolve()))
    wrist = torch.zeros(frames, 2, 6)
    theta = torch.zeros(frames, 2, 15)
    presence = torch.zeros(frames, 2, dtype=torch.bool)
    for frame, hands in enumerate(annotations):
        for slot, side in enumerate(("0", "1")):
            annotation = hands.get(side)
            if not annotation:
                continue
            wrist_data = annotation["wrist_xform"]
            rotation = quaternion_wxyz_to_matrix(torch.tensor(wrist_data["q_wxyz"]))
            wrist[frame, slot, :3] = matrix_to_axis_angle(rotation)
            wrist[frame, slot, 3:] = torch.tensor(wrist_data["t_xyz"])
            theta[frame, slot] = torch.tensor(annotation["pose"])
            presence[frame, slot] = True
    betas = shape[None].expand(frames, -1, -1).clone()
    global_world, articulation, root_world = [], [], []
    for slot, layer in enumerate((pca_left, pca_right)):
        result = layer(
            betas=betas[:, slot],
            global_orient=wrist[:, slot, :3],
            hand_pose=theta[:, slot],
            transl=wrist[:, slot, 3:],
            return_verts=True,
            return_full_pose=True,
        )
        global_world.append(axis_angle_to_matrix(result.full_pose[:, :3]))
        articulation.append(axis_angle_to_matrix(result.full_pose[:, 3:].reshape(frames, 15, 3)))
        root_world.append(result.joints[:, 0].float())
    global_world_tensor = torch.stack(global_world, dim=1)
    articulation_tensor = torch.stack(articulation, dim=1)
    root_world_tensor = torch.stack(root_world, dim=1)
    global_camera = torch.einsum("tij,tsjk->tsik", rotation_camera_from_world, global_world_tensor)
    root_camera = (
        torch.einsum("tij,tsj->tsi", rotation_camera_from_world, root_world_tensor)
        + translation_camera_from_world[:, None]
    )
    with torch.no_grad():
        joints_root, _ = canonical(global_camera, articulation_tensor, betas)
    return {
        "global_rotation": global_camera,
        "articulation": articulation_tensor,
        "betas": betas,
        "translation": root_camera,
        "joints_root": joints_root,
        "joints_camera": joints_root + root_camera.unsqueeze(-2),
        "presence": presence,
    }


def load_hot3d_window(
    recording_root: str | Path,
    start: int,
    frames: int,
    *,
    mano_model_path: str | Path,
    required_masks: tuple[str, ...] = REQUIRED_TRAINING_MASKS,
) -> DreamHandSample:
    """Decode one native-fisheye Aria RGB window with released MANO truth."""

    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        from projectaria_tools.core.stream_id import StreamId
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("HOT3D loading requires projectaria-tools") from error

    root = allowed_path(recording_root)
    timestamps = stream_timestamps(str(root / "masks/mask_hand_pose_available.csv"))
    window = timestamps[start : start + frames]
    if len(window) != frames:
        raise IndexError("HOT3D window exceeds the RGB timestamp index")
    for mask_name in required_masks:
        values = stream_mask(str(root / "masks" / f"{mask_name}.csv"))
        invalid = [timestamp for timestamp in window if not values.get(timestamp, False)]
        if invalid:
            raise ValueError(f"HOT3D window violates {mask_name}: {len(invalid)}/{frames} frames")
    hand_visible = stream_mask(str(root / "masks/mask_hand_visible.csv"))
    mano_poses, shape = _parse_mano(str(root / "mano_hand_pose_trajectory.jsonl"))
    headset, headset_times = _parse_headset(str(root / "headset_trajectory.csv"))
    provider = data_provider.create_vrs_data_provider(str(root / "recording.vrs"))
    if provider is None:
        raise RuntimeError(f"cannot open HOT3D VRS {root}")
    stream = StreamId(RGB_STREAM)
    label = provider.get_label_from_stream_id(stream)
    source_calibration = provider.get_device_calibration().get_camera_calib(label)
    source_width, source_height = source_calibration.get_image_size()
    source_size = torch.tensor([float(source_height), float(source_width)])
    camera_parameters = torch.from_numpy(
        np.asarray(source_calibration.get_projection_params(), dtype=np.float32)
    )
    if camera_parameters.shape != (15,):
        raise RuntimeError(
            f"HOT3D RGB camera is not the expected Fisheye624 model: {camera_parameters.shape}"
        )
    intrinsics = _upright_intrinsics(camera_parameters, source_size)
    gt_ray_field = _cached_ray_field(source_calibration, source_size)
    device_from_camera = torch.from_numpy(
        source_calibration.get_transform_device_camera().to_matrix()
    ).float()
    images, rotations, translations, annotations = [], [], [], []
    for timestamp in window:
        image_data = provider.get_image_data_by_time_ns(
            stream, timestamp, TimeDomain.TIME_CODE, TimeQueryOptions.CLOSEST
        )
        if image_data is None:
            raise RuntimeError(f"no HOT3D RGB frame near {timestamp}")
        images.append(_resize_native_image(image_data[0].to_numpy_array()))
        world_from_device = _closest(headset, headset_times, timestamp)
        rotation, translation = upright_camera_transform(world_from_device @ device_from_camera)
        rotations.append(rotation)
        translations.append(translation)
        annotations.append(mano_poses.get(timestamp, {}))
    geometry = _geometry(
        annotations,
        torch.stack(rotations),
        torch.stack(translations),
        str(mano_model_path),
        shape,
    )
    joints = geometry["joints_camera"]
    joints_2d = project_fisheye624_upright(
        joints.unsqueeze(0),
        camera_parameters.unsqueeze(0),
        source_size.unsqueeze(0),
        torch.tensor([[float(IMAGE_SIZE), float(IMAGE_SIZE)]]),
    ).squeeze(0)
    valid_joint = geometry["presence"].unsqueeze(-1).expand(-1, -1, 21)
    visible = valid_joint & (joints[..., 2] > 0.01)
    visible &= ((joints_2d >= 0) & (joints_2d < 1)).all(-1)
    visible &= torch.tensor([hand_visible.get(timestamp, False) for timestamp in window])[
        :, None, None
    ]
    video = torch.stack(images, dim=1).float().div(127.5).sub(1.0)
    sample = DreamHandSample(
        dataset="hot3d",
        recording_id=root.name,
        frame_indices=torch.arange(start, start + frames),
        video=video,
        intrinsics=intrinsics,
        image_size=torch.tensor([float(IMAGE_SIZE), float(IMAGE_SIZE)]),
        global_rotation=geometry["global_rotation"],
        articulation=geometry["articulation"],
        betas=geometry["betas"],
        translation=geometry["translation"],
        joints_root=geometry["joints_root"],
        joints_camera=joints,
        joints_2d=joints_2d,
        existence=geometry["presence"],
        visibility=visible.any(-1),
        valid_hand=geometry["presence"],
        valid_mano=geometry["presence"],
        valid_joints_3d=valid_joint,
        valid_joints_2d=visible,
        valid_ray=torch.tensor(True),
        gt_ray_field=gt_ray_field,
        distortion=torch.zeros(8),
        camera_model=FISHEYE624_UPRIGHT,
        camera_parameters=camera_parameters,
        source_image_size=source_size,
    )
    validate_sample(sample)
    return sample
