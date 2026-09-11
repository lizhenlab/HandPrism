"""Minimal LoRA injection matching all ten Wan linear layers per block."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TARGET_SUFFIXES = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 64,
        alpha: float = 64.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, value: Tensor) -> Tensor:
        update = F.linear(F.linear(self.dropout(value), self.lora_A), self.lora_B)
        return self.base(value) + update * self.scaling


@dataclass(frozen=True)
class LoRAReport:
    modules: int
    parameters: int
    names: tuple[str, ...]


def _get_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, (nn.ModuleList, nn.Sequential)) else getattr(parent, part)
    return parent, parts[-1]


def inject_wan_lora(
    model: nn.Module,
    rank: int = 64,
    alpha: float = 64.0,
    dropout: float = 0.0,
) -> LoRAReport:
    """Inject adapters only into `blocks.{0..29}` and fail on schema drift."""

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not name.startswith("blocks.") or not isinstance(module, nn.Linear):
            continue
        if name.endswith(TARGET_SUFFIXES):
            replacements.append((name, module))
    expected_blocks = len(getattr(model, "blocks", []))
    expected = expected_blocks * len(TARGET_SUFFIXES)
    if len(replacements) != expected:
        raise RuntimeError(
            f"official Wan module schema mismatch: found {len(replacements)} LoRA targets, expected {expected}"
        )
    names: list[str] = []
    count = 0
    for name, base in replacements:
        parent, child = _get_parent(model, name)
        wrapped = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child, wrapped)
        names.append(name)
        count += wrapped.lora_A.numel() + wrapped.lora_B.numel()
    return LoRAReport(modules=len(names), parameters=count, names=tuple(names))


def configure_trainable_backbone(model: nn.Module) -> None:
    """Freeze Wan except LoRA, patch embedding and the registered output head."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    if not hasattr(model, "patch_embedding"):
        raise RuntimeError("Wan model has no patch_embedding")
    for parameter in model.patch_embedding.parameters():
        parameter.requires_grad_(True)
    # Retain the output head in the optimizer/checkpoint schema. The feature
    # adapter stops at block 15, so this head receives no forward-pass gradient.
    if not hasattr(model, "head"):
        raise RuntimeError("Wan model has no diffusion output head")
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)


def promote_trainable_parameters(
    model: nn.Module, dtype: torch.dtype = torch.float32
) -> None:
    """Keep frozen Wan weights compact while giving optimized tensors FP32 masters."""

    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)


def lora_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B
