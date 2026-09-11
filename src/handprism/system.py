"""HandPrism trainable system receiving latents from the frozen VAE."""

from __future__ import annotations

from torch import Tensor, nn

from .camera import PINHOLE
from .backbone import WanCleanLatentEncoder
from .config import DecoderConfig, SolverConfig
from .model import HandPrismModel, HandPrismOutput, ManoProvider
from .fusion import FusionConfig


class HandPrismSystem(nn.Module):
    def __init__(
        self,
        encoder: WanCleanLatentEncoder,
        mano: ManoProvider,
        decoder_config: DecoderConfig = DecoderConfig(),
        solver_config: SolverConfig = SolverConfig(),
        *,
        architecture: str,
        fusion_config: FusionConfig = FusionConfig(),
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.architecture = architecture
        self.hand = HandPrismModel(
            mano, decoder_config, solver_config, architecture=architecture, fusion_config=fusion_config
        )

    def forward(
        self,
        clean_latent: Tensor,
        *,
        target_frames: int,
        solver: str,
        intrinsics: Tensor,
        image_size: Tensor,
        distortion: Tensor | None = None,
        calibration_ray_field: Tensor | None = None,
        camera_model: str = PINHOLE,
        camera_parameters: Tensor | None = None,
        source_image_size: Tensor | None = None,
        rgb_high: Tensor | None = None,
        roi_teacher: Tensor | None = None,
        roi_teacher_valid: Tensor | None = None,
        optimizer_step: int | None = None,
    ) -> HandPrismOutput:
        features = self.encoder(clean_latent)
        return self.hand(
            features,
            target_frames=target_frames,
            solver=solver,
            intrinsics=intrinsics,
            image_size=image_size,
            distortion=distortion,
            calibration_ray_field=calibration_ray_field,
            camera_model=camera_model,
            camera_parameters=camera_parameters,
            source_image_size=source_image_size,
            rgb_high=rgb_high, roi_teacher=roi_teacher,
            roi_teacher_valid=roi_teacher_valid, optimizer_step=optimizer_step,
        )
