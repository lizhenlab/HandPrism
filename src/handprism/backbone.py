"""Adapter for deterministic block-15 features from official VideoX-Fun Wan 2.2."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys
import types
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import BackboneConfig


class WanDependencyError(RuntimeError):
    pass


def _import_wan_module(source: Path):
    """Import only the audited Wan module, not VideoX-Fun's all-model registry.

    `videox_fun.models.__init__` eagerly imports unrelated audio/image models
    and their optional dependencies. Bypassing that registry keeps this
    adapter's dependency surface tied to the selected Wan backbone while
    loading the original module file unchanged.
    """

    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    importlib.import_module("videox_fun")
    # `wan_transformer3d.py` only needs this decorator from `videox_fun.utils`.
    # Importing that package normally pulls OpenCV and unrelated samplers.
    if "videox_fun.utils" not in sys.modules:
        utility_package = types.ModuleType("videox_fun.utils")

        def cfg_skip():
            def decorate(function):
                return function

            return decorate

        utility_package.cfg_skip = cfg_skip  # type: ignore[attr-defined]
        sys.modules["videox_fun.utils"] = utility_package
    package_name = "videox_fun.models"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source / "videox_fun" / "models")]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module("videox_fun.models.wan_transformer3d")


def load_official_wan(
    model_path: str | Path,
    videox_fun_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load the official low-noise DiT without silently accepting schema drift."""

    source = Path(videox_fun_path).resolve()
    if not (source / "videox_fun" / "models" / "wan_transformer3d.py").is_file():
        raise WanDependencyError(f"not a VideoX-Fun checkout: {source}")
    try:
        Wan2_2Transformer3DModel = _import_wan_module(source).Wan2_2Transformer3DModel
    except Exception as error:  # pragma: no cover - exercised on the GPU host
        raise WanDependencyError("failed to import the audited VideoX-Fun implementation") from error

    model_root = Path(model_path)
    candidates = [model_root / "transformer", model_root]
    last_error: Exception | None = None
    for candidate in candidates:
        if not (candidate / "config.json").is_file():
            continue
        try:
            model = Wan2_2Transformer3DModel.from_pretrained(
                str(candidate), torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )
            _validate_official_schema(model)
            return model
        except Exception as error:  # pragma: no cover - depends on downloaded snapshot
            last_error = error
    raise WanDependencyError(f"could not load Wan transformer from {model_root}") from last_error


def load_official_vae(
    checkpoint_path: str | Path,
    videox_fun_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    source = Path(videox_fun_path).resolve()
    if not (source / "videox_fun" / "models" / "wan_vae3_8.py").is_file():
        raise WanDependencyError(f"not a VideoX-Fun checkout: {source}")
    _import_wan_module(source)
    module = importlib.import_module("videox_fun.models.wan_vae3_8")
    vae = module.AutoencoderKLWan3_8.from_pretrained(str(checkpoint_path))
    return vae.to(dtype=torch_dtype).eval().requires_grad_(False)


class WanFrozenVAEEncoder(nn.Module):
    """Frozen deterministic VAE mode used to create the clean 48-channel latent."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, video: Tensor) -> Tensor:
        """Encode RGB `[B,3,T,H,W]` normalized to `[-1,1]`."""

        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must be [B,3,T,H,W]")
        if video.amin() < -1.001 or video.amax() > 1.001:
            raise ValueError("video must be normalized to [-1,1]")
        distribution = self.vae.encode(video).latent_dist
        latent = distribution.mode()
        expected_frames = (video.shape[2] - 1) // 4 + 1
        if latent.shape[1] != 48 or latent.shape[2] != expected_frames:
            raise RuntimeError(
                f"unexpected Wan VAE latent shape {tuple(latent.shape)}; expected 48 channels and {expected_frames} frames"
            )
        return latent


def _validate_official_schema(model: nn.Module, config: BackboneConfig = BackboneConfig()) -> None:
    checks = {
        "dim": config.feature_dim,
        "ffn_dim": config.ffn_dim,
        "num_layers": config.layers,
        "in_dim": config.input_channels,
    }
    actual_config: Any = getattr(model, "config", None)
    for key, expected in checks.items():
        actual = getattr(actual_config, key, None)
        if actual is None and hasattr(actual_config, "get"):
            actual = actual_config.get(key)
        if actual != expected:
            raise RuntimeError(f"Wan config {key}={actual!r}; audited value is {expected!r}")
    if len(model.blocks) != config.layers:
        raise RuntimeError("Wan block count differs from the audited release")
    weight = model.patch_embedding.weight
    if tuple(weight.shape) != (3072, 148, 1, 2, 2):
        raise RuntimeError(f"unexpected patch embedding shape: {tuple(weight.shape)}")


class WanCleanLatentEncoder(nn.Module):
    """Execute the official Wan preamble and blocks 0..15, without its output head."""

    def __init__(
        self,
        model: nn.Module,
        config: BackboneConfig = BackboneConfig(),
        *,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        _validate_official_schema(model, config)
        self.model = model
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

    def _sinusoidal_embedding(self, timestep: Tensor) -> Tensor:
        try:
            module = sys.modules["videox_fun.models.wan_transformer3d"]
            sinusoidal_embedding_1d = module.sinusoidal_embedding_1d
        except Exception as error:  # pragma: no cover
            raise WanDependencyError("VideoX-Fun is not importable") from error
        return sinusoidal_embedding_1d(self.model.freq_dim, timestep)

    def forward(
        self,
        clean_latent: Tensor,
        *,
        condition: Tensor | None = None,
        context: Tensor | None = None,
    ) -> Tensor:
        """Return `[B,T',H',W',3072]` tapped features.

        Absent conditioning uses 100 zero channels and a zero 512x4096 text
        context, as configured by BackboneConfig. No text prompt is inferred.
        """

        if clean_latent.ndim != 5 or clean_latent.shape[1] != self.config.latent_channels:
            raise ValueError("clean_latent must be [B,48,T,H,W]")
        batch, _, frames, _, _ = clean_latent.shape
        if condition is None:
            condition = clean_latent.new_zeros(
                batch,
                self.config.condition_channels,
                *clean_latent.shape[2:],
            )
        expected_condition = (batch, self.config.condition_channels, *clean_latent.shape[2:])
        if tuple(condition.shape) != expected_condition:
            raise ValueError(f"condition must be {expected_condition}")
        if context is None:
            context = clean_latent.new_zeros(
                batch, self.config.text_length, self.config.text_dim
            )
        if context.ndim != 3 or context.shape[0] != batch or context.shape[-1] != self.config.text_dim:
            raise ValueError("context must be [B,L,4096]")
        if context.shape[1] > self.config.text_length:
            raise ValueError("context exceeds the official 512-token limit")
        if getattr(self.model, "sp_world_size", 1) != 1:
            raise RuntimeError("sequence-parallel Wan is outside this adapter's audited path")

        dtype = clean_latent.dtype
        device = self.model.patch_embedding.weight.device
        if self.model.freqs.device != device and device.type != "meta":
            self.model.freqs = self.model.freqs.to(device)
        packed = torch.cat((clean_latent, condition), dim=1)
        patched = self.model.patch_embedding(packed)
        grid = torch.tensor(patched.shape[2:], dtype=torch.long, device=patched.device)
        grid_sizes = grid.unsqueeze(0).expand(batch, -1).contiguous()
        sequence = patched.flatten(2).transpose(1, 2)
        seq_len = sequence.shape[1]
        seq_lens = torch.full((batch,), seq_len, dtype=torch.long, device=sequence.device)

        timestep = torch.zeros(batch, device=sequence.device, dtype=torch.float32)
        time_input = self._sinusoidal_embedding(timestep).to(
            dtype=self.model.time_embedding[0].weight.dtype
        )
        time_embed = self.model.time_embedding(time_input).to(dtype)
        modulation = self.model.time_projection(time_embed).unflatten(1, (6, self.model.dim))
        if context.shape[1] < self.config.text_length:
            context = torch.cat(
                (
                    context,
                    context.new_zeros(batch, self.config.text_length - context.shape[1], context.shape[2]),
                ),
                dim=1,
            )
        embedded_context = self.model.text_embedding(context)

        for index, block in enumerate(self.model.blocks):
            if index > self.config.tap_block:
                break

            def run_block(value: Tensor, layer: nn.Module = block) -> Tensor:
                return layer(
                    value,
                    e=modulation,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=self.model.freqs,
                    context=embedded_context,
                    context_lens=None,
                    dtype=dtype,
                    t=timestep,
                )

            if self.gradient_checkpointing and torch.is_grad_enabled():
                sequence = checkpoint(run_block, sequence, use_reentrant=False)
            else:
                sequence = run_block(sequence)

        grid_t, grid_h, grid_w = (int(item) for item in grid.tolist())
        if grid_t != frames:
            raise RuntimeError("audited patch embedding must preserve latent time")
        return sequence.reshape(batch, grid_t, grid_h, grid_w, self.config.feature_dim)
