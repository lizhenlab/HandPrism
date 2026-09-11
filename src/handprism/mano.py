"""MANO boundary and a synthetic test double.

The licensed MANO parameter files are never redistributed. The SMPL-X adapter
follows the standard manopth fingertip indices and OpenPose 21-joint ordering.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .precision import fp32_geometry


@torch.no_grad()
def correct_left_shapedirs(left: nn.Module, right: nn.Module) -> bool:
    """Apply the HOT3D/SMPL-X left-hand asset correction exactly once."""

    if getattr(left, "_handprism_shapedirs_checked", False):
        return False
    needs_fix = bool(torch.sum(torch.abs(left.shapedirs[:, 0] - right.shapedirs[:, 0])) < 1)
    if needs_fix:
        left.shapedirs[:, 0].mul_(-1)
    left._handprism_shapedirs_checked = True
    return needs_fix


def matrix_to_axis_angle(matrix: Tensor, eps: float = 1e-8) -> Tensor:
    """Stable rotation-matrix to axis-angle conversion through quaternions."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must end in [3,3]")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    squared = torch.stack(
        (
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ),
        dim=-1,
    )
    absolute = torch.where(squared > 0, squared.clamp_min(eps).sqrt(), torch.zeros_like(squared))
    candidates = torch.stack(
        (
            torch.stack((absolute[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, absolute[..., 1].square(), m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, absolute[..., 2].square(), m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, absolute[..., 3].square()), -1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * absolute.clamp_min(0.1)).unsqueeze(-1)
    choice = absolute.argmax(-1)
    quaternion = candidates.gather(-2, choice[..., None, None].expand(*choice.shape, 1, 4)).squeeze(
        -2
    )
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1, eps=eps)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    norm = vector.norm(dim=-1, keepdim=True)
    scale = torch.where(
        norm > eps,
        2.0 * torch.atan2(norm, quaternion[..., :1]) / norm.clamp_min(eps),
        torch.full_like(norm, 2.0),
    )
    return vector * scale


class ToyMano(nn.Module):
    """Differentiable shape-compatible test double; never use for reported metrics."""

    def __init__(self) -> None:
        super().__init__()
        joints = [[0.0, 0.0, 0.0]]
        for finger in range(5):
            for level in range(1, 5):
                joints.append([(finger - 2) * 0.018, level * 0.025, -0.002 * abs(finger - 2)])
        self.register_buffer("rest_joints", torch.tensor(joints, dtype=torch.float32))

    def root_offset(self, betas: Tensor) -> Tensor:
        return betas.new_zeros(*betas.shape[:-1], 3)

    @fp32_geometry
    def forward(
        self, global_rotation: Tensor, articulation: Tensor, betas: Tensor
    ) -> tuple[Tensor, Tensor]:
        leading = global_rotation.shape[:-2]
        rest = self.rest_joints.view((1,) * len(leading) + (21, 3)).expand(*leading, 21, 3).clone()
        side = rest.new_tensor([-1.0, 1.0]).view((1,) * (len(leading) - 1) + (2, 1, 1))
        rest[..., 0] = rest[..., 0] * side[..., 0]
        rest = rest * (1.0 + 0.03 * betas[..., :1]).unsqueeze(-1)
        indices = torch.arange(20, device=rest.device) % 15
        fingers = (articulation.index_select(-3, indices) @ rest[..., 1:, :, None]).squeeze(-1)
        posed = torch.cat((rest[..., :1, :], fingers), dim=-2)
        joints = (global_rotation.unsqueeze(-3) @ posed.unsqueeze(-1)).squeeze(-1)
        vertices = joints.index_select(-2, torch.arange(778, device=joints.device) % 21)
        return joints, vertices


class SmplxMano(nn.Module):
    _OPENPOSE_ORDER = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
    _RIGHT_TIPS = (745, 317, 444, 556, 673)
    _LEFT_TIPS = (745, 317, 445, 556, 673)

    def __init__(self, model_path: str | Path, *, flat_hand_mean: bool = True) -> None:
        super().__init__()
        # Licensed MANO pkl files can contain legacy scipy/chumpy objects.
        # Restore removed inspection/NumPy names only during compatible load.
        import inspect
        import numpy as np

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
        for name, value in {
            "bool": bool,
            "int": int,
            "float": float,
            "complex": complex,
            "object": object,
            "unicode": str,
            "str": str,
        }.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        try:
            import smplx
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("SmplxMano requires `pip install smplx`") from error
        root = Path(model_path)
        if root.name.lower() == "mano":
            root = root.parent
        common = dict(
            model_path=str(root),
            model_type="mano",
            use_pca=False,
            flat_hand_mean=flat_hand_mean,
            num_betas=10,
            create_global_orient=False,
            create_hand_pose=False,
            create_betas=False,
            create_transl=False,
        )
        self.left = smplx.create(is_rhand=False, **common)
        self.right = smplx.create(is_rhand=True, **common)
        correct_left_shapedirs(self.left, self.right)

    @fp32_geometry
    def root_offset(self, betas: Tensor) -> Tensor:
        """Untranslated MANO J0, so native translation = wrist_camera - J0.

        SMPL-X rotates about the shaped rest joint, not about the origin.
        Its untranslated root is therefore shape-dependent but pose-invariant.
        """

        if betas.shape[-2:] != (2, 10):
            raise ValueError("betas must end in [left/right,10]")
        roots = []
        for slot, layer in enumerate((self.left, self.right)):
            regressor = layer.J_regressor[0].float()
            template_root = torch.einsum("v,vc->c", regressor, layer.v_template.float())
            shape_root = torch.einsum("v,vck->ck", regressor, layer.shapedirs.float())
            roots.append(
                template_root + torch.einsum("...k,ck->...c", betas[..., slot, :], shape_root)
            )
        return torch.stack(roots, dim=-2)

    def _side(
        self,
        layer: nn.Module,
        global_rotation: Tensor,
        articulation: Tensor,
        betas: Tensor,
        tips: tuple[int, ...],
    ) -> tuple[Tensor, Tensor]:
        layer_dtype = layer.shapedirs.dtype
        result = layer(
            global_orient=matrix_to_axis_angle(global_rotation.to(layer_dtype)),
            hand_pose=matrix_to_axis_angle(articulation.to(layer_dtype)).flatten(-2),
            betas=betas.to(layer_dtype),
            return_verts=True,
        )
        vertices = result.vertices
        joints = torch.cat((result.joints[:, :16], vertices[:, list(tips)]), dim=1)
        joints = joints[:, list(self._OPENPOSE_ORDER)]
        root = joints[:, :1]
        return joints - root, vertices - root

    @fp32_geometry
    def forward(
        self, global_rotation: Tensor, articulation: Tensor, betas: Tensor
    ) -> tuple[Tensor, Tensor]:
        if global_rotation.shape[-3] != 2:
            raise ValueError("hand slots must be ordered [left,right]")
        leading = global_rotation.shape[:-3]
        all_joints, all_vertices = [], []
        for slot, (layer, tips) in enumerate(
            ((self.left, self._LEFT_TIPS), (self.right, self._RIGHT_TIPS))
        ):
            joints, vertices = self._side(
                layer,
                global_rotation[..., slot, :, :].reshape(-1, 3, 3),
                articulation[..., slot, :, :, :].reshape(-1, 15, 3, 3),
                betas[..., slot, :].reshape(-1, 10),
                tips,
            )
            all_joints.append(joints.reshape(*leading, 21, 3))
            all_vertices.append(vertices.reshape(*leading, 778, 3))
        return torch.stack(all_joints, dim=-3), torch.stack(all_vertices, dim=-3)
