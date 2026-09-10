from __future__ import annotations

import torch
from torch import nn

from dreamhand.lora import LoRALinear, inject_wan_lora, promote_trainable_parameters


class Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, width)


class Block(nn.Module):
    def __init__(self, width: int, ffn: int) -> None:
        super().__init__()
        self.self_attn = Attention(width)
        self.cross_attn = Attention(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))


class FakeWan(nn.Module):
    def __init__(self, blocks: int, width: int, ffn: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(width, ffn) for _ in range(blocks)])


def test_lora_injects_ten_modules_per_block_and_starts_as_identity() -> None:
    model = FakeWan(blocks=2, width=16, ffn=48)
    value = torch.randn(3, 16)
    before = model.blocks[0].self_attn.q(value).detach()
    report = inject_wan_lora(model, rank=4, alpha=4)
    assert report.modules == 20
    expected_per_block = 8 * 4 * (16 + 16) + 2 * 4 * (16 + 48)
    assert report.parameters == 2 * expected_per_block
    assert isinstance(model.blocks[0].self_attn.q, LoRALinear)
    torch.testing.assert_close(model.blocks[0].self_attn.q(value), before)


def test_wan_lora_parameter_arithmetic() -> None:
    per_block = 8 * 64 * (3072 + 3072) + 2 * 64 * (3072 + 14336)
    assert per_block == 5_373_952
    assert per_block * 30 == 161_218_560
    assert per_block * 16 == 85_983_232


def test_trainable_lora_has_fp32_master_while_base_stays_bfloat16() -> None:
    layer = LoRALinear(nn.Linear(8, 8).to(torch.bfloat16), rank=2, alpha=2)
    promote_trainable_parameters(layer)
    assert layer.base.weight.dtype == torch.bfloat16
    assert layer.lora_A.dtype == torch.float32
    assert layer.lora_B.dtype == torch.float32
