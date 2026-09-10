"""End-to-end assembly around a pluggable MANO implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch import Tensor, nn

from .camera import PINHOLE, project_camera
from .architectures import architecture_spec
from .config import DecoderConfig, SolverConfig
from .decoder import DreamHandDecoder, DreamHandDecoderOutput
from .precision import fp32_geometry
from .ray import (
    EffectivePinholeCamera,
    MixedPnPOutput,
    RayHead,
    kfree_bearings,
    bearings_from_intrinsics,
    mixed_pnp,
    project_kfree,
    sample_ray_bearings,
)


class ManoProvider(Protocol):
    def root_offset(self, betas: Tensor) -> Tensor:
        """Return untranslated MANO J0 in the camera-oriented model frame."""

    def __call__(
        self,
        global_rotation: Tensor,
        articulation: Tensor,
        betas: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return root-centered joints `[B,T,2,21,3]` and vertices."""


@dataclass
class DreamHandOutput:
    decoder: DreamHandDecoderOutput
    ray_field: Tensor
    joints_root_mano: Tensor
    vertices_root: Tensor
    joints_camera: Tensor
    vertices_camera: Tensor
    pnp: MixedPnPOutput
    camera_fit: EffectivePinholeCamera | None
    mano_translation: Tensor


class DreamHandModel(nn.Module):
    def __init__(
        self,
        mano: ManoProvider,
        decoder_config: DecoderConfig = DecoderConfig(),
        solver_config: SolverConfig = SolverConfig(),
        *,
        architecture: str,
    ) -> None:
        super().__init__()
        architecture_spec(architecture)
        self.architecture = architecture
        self.decoder = DreamHandDecoder(decoder_config, architecture=architecture)
        self.ray_head = RayHead(decoder_config.feature_dim)
        self.mano = mano
        self.solver_config = replace(solver_config, architecture=architecture)

    def forward(
        self,
        features: Tensor,
        *,
        target_frames: int,
        solver: str,
        intrinsics: Tensor | None = None,
        image_size: Tensor,
        distortion: Tensor | None = None,
        calibration_ray_field: Tensor | None = None,
        camera_model: str = PINHOLE,
        camera_parameters: Tensor | None = None,
        source_image_size: Tensor | None = None,
    ) -> DreamHandOutput:
        ray_field = self.ray_head(features)
        decoded = self.decoder(features, ray_field, target_frames)
        return self._geometry(
            decoded,
            ray_field,
            target_frames=target_frames,
            solver=solver,
            intrinsics=intrinsics,
            image_size=image_size,
            distortion=distortion,
            calibration_ray_field=calibration_ray_field,
            camera_model=camera_model,
            camera_parameters=camera_parameters,
            source_image_size=source_image_size,
        )

    @fp32_geometry
    def _geometry(
        self,
        decoded: DreamHandDecoderOutput,
        ray_field: Tensor,
        *,
        target_frames: int,
        solver: str,
        intrinsics: Tensor | None,
        image_size: Tensor,
        distortion: Tensor | None,
        calibration_ray_field: Tensor | None,
        camera_model: str,
        camera_parameters: Tensor | None,
        source_image_size: Tensor | None,
    ) -> DreamHandOutput:
        betas = decoded.betas[:, None].expand(-1, target_frames, -1, -1)
        joints, vertices = self.mano(
            decoded.global_rotation,
            decoded.articulation,
            betas,
        )
        camera_fit: EffectivePinholeCamera | None = None
        if solver == "standard":
            if intrinsics is None:
                raise ValueError("standard solver requires camera calibration")
            if calibration_ray_field is not None:
                bearings = sample_ray_bearings(calibration_ray_field, decoded.anchors_2d)
            else:
                if intrinsics is None:
                    raise ValueError("standard solver requires camera calibration")
                bearings = bearings_from_intrinsics(
                    decoded.anchors_2d, intrinsics, image_size, distortion
                )

            def projector(points: Tensor) -> tuple[Tensor, Tensor]:
                uv = project_camera(
                    points,
                    intrinsics,
                    image_size,
                    distortion,
                    camera_model=camera_model,
                    camera_parameters=camera_parameters,
                    source_image_size=source_image_size,
                )
                return uv, torch.isfinite(uv).all(-1) & (points[..., 2] > 0)
        elif solver == "kfree":
            # Fit a clip-level effective camera to predicted rays. The selected
            # architecture controls fit acceptance; rejected fits use direct
            # sampling for both mixed-PnP and the wrist fallback.
            bearings, camera_fit = kfree_bearings(
                ray_field,
                decoded.anchors_2d,
                self.solver_config,
            )

            def projector(points: Tensor) -> tuple[Tensor, Tensor]:
                return project_kfree(
                    points, camera_fit, ray_field, decoded.anchors_2d, self.solver_config
                )
        else:
            raise ValueError("solver must be 'standard' or 'kfree'")
        pnp = mixed_pnp(
            joints,
            decoded.anchors_2d,
            decoded.log_depth,
            bearings,
            image_size,
            self.solver_config,
            projector=projector,
        )
        joints_camera = joints + pnp.translation.unsqueeze(-2)
        vertices_camera = vertices + pnp.translation.unsqueeze(-2)
        return DreamHandOutput(
            decoder=decoded,
            ray_field=ray_field,
            joints_root_mano=joints,
            vertices_root=vertices,
            joints_camera=joints_camera,
            vertices_camera=vertices_camera,
            pnp=pnp,
            camera_fit=camera_fit,
            mano_translation=pnp.translation - self.mano.root_offset(betas),
        )
