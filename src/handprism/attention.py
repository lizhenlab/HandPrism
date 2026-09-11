"""Cross-attention and bidirectional temporal/query attention for HandPrism."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _split_heads(value: Tensor, heads: int) -> Tensor:
    batch, length, width = value.shape
    if width % heads:
        raise ValueError("hidden width must be divisible by attention heads")
    return value.view(batch, length, heads, width // heads).transpose(1, 2)


def _merge_heads(value: Tensor) -> Tensor:
    batch, heads, length, head_dim = value.shape
    return value.transpose(1, 2).contiguous().view(batch, length, heads * head_dim)


def _rope(value: Tensor, positions: Tensor) -> Tensor:
    """Apply temporal rotary embeddings to `[B,H,T,Dh]`."""

    head_dim = value.shape[-1]
    rotary_dim = head_dim - head_dim % 2
    if rotary_dim == 0:
        return value
    inv_freq = 1.0 / (
        10_000
        ** (torch.arange(0, rotary_dim, 2, device=value.device, dtype=torch.float32) / rotary_dim)
    )
    phase = torch.outer(positions.to(torch.float32), inv_freq).to(value.dtype)
    cos = phase.cos()[None, None]
    sin = phase.sin()[None, None]
    rotary, tail = value[..., :rotary_dim], value[..., rotary_dim:]
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rotated = rotated.flatten(-2)
    return torch.cat((rotated, tail), dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, width)

    def forward(self, query: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        q = _split_heads(self.q(query), self.heads)
        k = _split_heads(self.k(memory), self.heads)
        v = _split_heads(self.v(memory), self.heads)
        scale = 1.0 / math.sqrt(q.shape[-1])
        # These probabilities also encode sub-pixel coordinates. Keep both
        # score accumulation and the exported heatmap out of BF16 autocast.
        with torch.autocast(query.device.type, enabled=False):
            logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
            weights = logits.softmax(dim=-1)
        if self.training and self.dropout:
            attended = torch.matmul(F.dropout(weights, p=self.dropout).to(v.dtype), v)
        else:
            attended = torch.matmul(weights.to(v.dtype), v)
        return self.o(_merge_heads(attended)), weights.mean(dim=1)


class TemporalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, width)

    def forward(self, value: Tensor, positions: Tensor | None = None) -> Tensor:
        if positions is None:
            positions = torch.arange(value.shape[1], device=value.device)
        if positions.shape != (value.shape[1],):
            raise ValueError("one temporal position is required per query token")
        q = _rope(_split_heads(self.q(value), self.heads), positions)
        k = _rope(_split_heads(self.k(value), self.heads), positions)
        v = _split_heads(self.v(value), self.heads)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.o(_merge_heads(attended))


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class AlternatingLayer(nn.Module):
    """Spatial read followed by bidirectional time/query self-attention.

    With joint_time_query=True, queries including registers can communicate
    across query slots; otherwise each slot only attends along time. Neither
    branch writes to visual memory. RoPE encodes frame time, not slot index.
    """

    def __init__(
        self, width: int, heads: int, ffn_dim: int, dropout: float = 0.0,
        *, joint_time_query: bool = True,
    ) -> None:
        super().__init__()
        self.joint_time_query = joint_time_query
        self.spatial_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.spatial = CrossAttention(width, heads, dropout)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal = TemporalSelfAttention(width, heads, dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, ffn_dim, dropout)

    def forward(self, queries: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        """Inputs are `[B,T,Q,D]` and `[B,T,S,D]`."""

        batch, frames, query_count, width = queries.shape
        spatial_update, weights = self.spatial(
            self.spatial_norm(queries).reshape(batch * frames, query_count, width),
            self.memory_norm(memory).reshape(batch * frames, memory.shape[2], width),
        )
        queries = queries + spatial_update.reshape(batch, frames, query_count, width)
        if self.joint_time_query:
            temporal_input = self.temporal_norm(queries).reshape(batch, frames * query_count, width)
            positions = torch.arange(frames, device=queries.device).repeat_interleave(query_count)
            temporal_update = self.temporal(temporal_input, positions).reshape(
                batch, frames, query_count, width
            )
        else:
            temporal_input = self.temporal_norm(queries).permute(0, 2, 1, 3).reshape(
                batch * query_count, frames, width
            )
            temporal_update = self.temporal(temporal_input).reshape(
                batch, query_count, frames, width
            ).permute(0, 2, 1, 3)
        queries = queries + temporal_update
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries, weights.reshape(batch, frames, query_count, memory.shape[2])
