"""Clip-coherent appearance augmentation without changing camera geometry."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F

from .contract import HandPrismSample


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    color_strength: float = .15
    blur_probability: float = .2
    jpeg_probability: float = .2
    occlusion_probability: float = .25
    resolution_min_scale: float = .7

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("augmentation.enabled must be boolean")
        for key in ("color_strength", "blur_probability", "jpeg_probability", "occlusion_probability"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        if not 0 < self.resolution_min_scale <= 1:
            raise ValueError("resolution_min_scale must be in (0,1]")


def augment_sample(sample: HandPrismSample, config: AugmentationConfig, seed: int) -> HandPrismSample:
    if not config.enabled:
        return sample
    generator = torch.Generator().manual_seed(seed)
    random = torch.rand(12, generator=generator).tolist()
    gain = 1 + (random[0] * 2 - 1) * config.color_strength
    bias = (random[1] * 2 - 1) * config.color_strength
    blur = random[2] < config.blur_probability
    jpeg = random[3] < config.jpeg_probability
    quality = int(40 + random[4] * 50)
    scale = config.resolution_min_scale + random[5] * (1 - config.resolution_min_scale)
    occlude = random[6] < config.occlusion_probability
    x, y = random[7] * .8, random[8] * .8
    w, h = .1 + random[9] * .2, .1 + random[10] * .2

    def apply(video: torch.Tensor) -> torch.Tensor:
        uint8 = video.dtype == torch.uint8
        result = []
        for frame in video.unbind(1):
            image = frame.float() / 127.5 - 1 if uint8 else frame
            image = (image * gain + bias).clamp(-1, 1)
            size = image.shape[-2:]
            small = tuple(max(1, round(value * scale)) for value in size)
            image = F.interpolate(image[None], size=small, mode="bilinear", align_corners=False, antialias=True)
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)[0]
            if blur:
                image = F.avg_pool2d(F.pad(image[None], (1, 1, 1, 1), mode="replicate"), 3, 1)[0]
            if jpeg:
                from PIL import Image
                buffer = BytesIO()
                array = ((image + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
                Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    image = torch.from_numpy(np.asarray(decoded).copy()).permute(2, 0, 1).float() / 127.5 - 1
            if occlude:
                image[:, int(y * size[0]):int(min(y+h, 1) * size[0]),
                      int(x * size[1]):int(min(x+w, 1) * size[1])] = 0.
            result.append(((image + 1) * 127.5).round().byte() if uint8 else image)
        return torch.stack(result, 1)

    sample.video = apply(sample.video)
    if sample.rgb_high is not None:
        sample.rgb_high = apply(sample.rgb_high)
    uv = sample.joints_2d
    mask = ((uv[..., 0] >= x) & (uv[..., 0] < min(x+w, 1)) &
            (uv[..., 1] >= y) & (uv[..., 1] < min(y+h, 1)))
    sample.synthetic_occluded = mask & occlude & sample.valid_joints_2d.bool()
    # Keep 3D supervision and geometric 2D targets on occluded hands; only the
    # separately calibrated observation-quality target is reduced.
    return sample
