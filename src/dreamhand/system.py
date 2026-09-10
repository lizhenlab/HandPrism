"""HandPrism trainable system receiving latents from the frozen VAE."""

from __future__ import annotations

from torch import Tensor, nn

from .camera import PINHOLE
from .backbone import WanCleanLatentEncoder
from .config import DecoderConfig, SolverConfig
from .model import DreamHandModel, DreamHandOutput, ManoProvider


class DreamHandSystem(nn.Module):
    def __init__(
        self,
        encoder: WanCleanLatentEncoder,
        mano: ManoProvider,
        decoder_config: DecoderConfig = DecoderConfig(),
        solver_config: SolverConfig = SolverConfig(),
        *,
        architecture: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.architecture = architecture
        self.hand = DreamHandModel(
            mano, decoder_config, solver_config, architecture=architecture
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
    ) -> DreamHandOutput:
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
        )
