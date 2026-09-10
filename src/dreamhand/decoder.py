"""Explicit Core/Fusion spatiotemporal decoders for HandPrism."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .attention import AlternatingLayer
from .architectures import CORE, FUSION, architecture_spec
from .config import DecoderConfig
from .positional import LearnedSpatialPE, RayPE, normalized_cell_centers
from .rotations import rotation_6d_to_matrix
from .precision import fp32_geometry


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
    )


def interpolate_time(value: Tensor, target_frames: int) -> Tensor:
    """Linear interpolation of `[B,T,...]`, preserving all trailing axes."""

    if value.shape[1] == target_frames:
        return value
    batch, frames = value.shape[:2]
    trailing = value.shape[2:]
    flat = value.reshape(batch, frames, -1).transpose(1, 2)
    flat = F.interpolate(flat, size=target_frames, mode="linear", align_corners=True)
    return flat.transpose(1, 2).reshape(batch, target_frames, *trailing)


@dataclass
class DreamHandDecoderOutput:
    query_features_latent: Tensor
    hand_features: Tensor
    joint_features: Tensor
    global_rotation_6d: Tensor
    global_rotation: Tensor
    articulation_6d: Tensor
    articulation: Tensor
    betas: Tensor
    log_depth: Tensor
    depth: Tensor
    existence_logits: Tensor
    visibility_logits: Tensor
    joints_root_direct: Tensor
    anchors_2d: Tensor
    anchors_2d_latent: Tensor
    attention_heatmaps: Tensor


class DreamHandDecoder(nn.Module):
    def __init__(self, config: DecoderConfig = DecoderConfig(), *, architecture: str) -> None:
        super().__init__()
        architecture_spec(architecture)
        self.architecture = architecture
        if architecture == CORE and config.anchor_offset_cells != 0:
            raise ValueError("HandPrism-Core does not support the Fusion anchor offset head")
        self.config = config
        width = config.hidden_dim
        self.feature_projection = nn.Linear(config.feature_dim, width)
        self.feature_norm = nn.LayerNorm(width)
        self.spatial_pe = LearnedSpatialPE(width, config.spatial_pe_height, config.spatial_pe_width)
        self.ray_pe = RayPE(width, config.ray_frequencies)
        self.queries = nn.Parameter(torch.empty(config.queries, width))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList(
            [
                AlternatingLayer(
                    width, config.heads, config.ffn_dim, config.dropout,
                    joint_time_query=architecture == FUSION,
                )
                for _ in range(config.layers)
            ]
        )
        pose_dim = 6 + config.mano_pose_joints * 6
        self.pose_head = _mlp(width, width, pose_dim)
        self.shape_head = _mlp(width, width, config.mano_shape_dim)
        self.camera_head = _mlp(width, width, 1)
        self.existence_head = _mlp(width, width, 1)
        self.visibility_head = _mlp(width, width, 1)
        self.joint_head = _mlp(width, width, 3)
        self.anchor_offset_head = nn.Linear(width, 2) if config.anchor_offset_cells > 0 else None
        if self.anchor_offset_head is not None:
            nn.init.zeros_(self.anchor_offset_head.weight)
            nn.init.zeros_(self.anchor_offset_head.bias)

    def forward(
        self,
        features: Tensor,
        ray_field: Tensor,
        target_frames: int,
    ) -> DreamHandDecoderOutput:
        """Decode `[B,T',H,W,D]` (default D=3072) and rays to RGB-frame outputs."""

        if features.ndim != 5:
            raise ValueError("features must have shape [B,T,H,W,D]")
        batch, latent_frames, height, width, channels = features.shape
        if channels != self.config.feature_dim:
            raise ValueError(f"expected feature width {self.config.feature_dim}, got {channels}")
        if ray_field.shape != (batch, height, width, 3):
            raise ValueError("ray field must match the feature grid")

        memory = self.feature_norm(self.feature_projection(features))
        memory = memory.reshape(batch, latent_frames, height * width, self.config.hidden_dim)
        spatial = self.spatial_pe(height, width)[:, None]
        ray = self.ray_pe(ray_field).reshape(batch, 1, height * width, self.config.hidden_dim)
        memory = memory + spatial + ray
        queries = self.queries[None, None].expand(batch, latent_frames, -1, -1)

        weights: Tensor | None = None
        for layer in self.layers:
            queries, weights = layer(queries, memory)
        assert weights is not None

        return self._readout(queries, weights, target_frames, height, width)

    @fp32_geometry
    def _readout(
        self, queries: Tensor, weights: Tensor, target_frames: int, height: int, width: int
    ) -> DreamHandDecoderOutput:
        batch, latent_frames = queries.shape[:2]

        hand_end = self.config.hand_queries
        joint_end = hand_end + self.config.joint_queries
        hand_latent = queries[:, :, :hand_end]
        joint_latent = queries[:, :, hand_end:joint_end]
        joint_latent = joint_latent.reshape(
            batch,
            latent_frames,
            self.config.hand_queries,
            self.config.joints_per_hand,
            self.config.hidden_dim,
        )
        hand = interpolate_time(hand_latent, target_frames)
        joints = interpolate_time(joint_latent, target_frames)

        pose = self.pose_head(hand)
        global_6d = pose[..., :6]
        articulation_6d = pose[..., 6:].reshape(
            batch,
            target_frames,
            self.config.hand_queries,
            self.config.mano_pose_joints,
            6,
        )
        shape_latent = hand_latent.mean(dim=1)
        betas = self.shape_head(shape_latent)
        log_depth = self.camera_head(hand)
        existence = self.existence_head(hand).squeeze(-1)
        visibility = self.visibility_head(hand).squeeze(-1)
        # Core applies the joint head after feature interpolation; Fusion
        # applies it at latent times before interpolating coordinates.
        if self.architecture == CORE:
            joints_root = self.joint_head(joints)
            joints_root = joints_root - joints_root[..., :1, :]
        else:
            joints_root_latent = self.joint_head(joint_latent)
            joints_root_latent = joints_root_latent - joints_root_latent[..., :1, :]
            joints_root = interpolate_time(joints_root_latent, target_frames)

        joint_weights = weights[:, :, hand_end:joint_end]
        joint_weights = joint_weights.reshape(
            batch,
            latent_frames,
            self.config.hand_queries,
            self.config.joints_per_hand,
            height * width,
        )
        centers = normalized_cell_centers(
            height, width, device=queries.device, dtype=queries.dtype
        ).reshape(height * width, 2)
        anchors_latent = torch.einsum("btsjn,nc->btsjc", joint_weights, centers)
        if self.anchor_offset_head is not None:
            cell_size = anchors_latent.new_tensor([1.0 / width, 1.0 / height])
            offset = self.anchor_offset_head(joint_latent).tanh()
            anchors_latent = anchors_latent + offset * cell_size * self.config.anchor_offset_cells
            # Do not clamp here: out-of-image predictions must remain visible
            # to the solver guards and to existence/visibility diagnostics.
        anchors = interpolate_time(anchors_latent, target_frames)

        return DreamHandDecoderOutput(
            query_features_latent=queries,
            hand_features=hand,
            joint_features=joints,
            global_rotation_6d=global_6d,
            global_rotation=rotation_6d_to_matrix(global_6d),
            articulation_6d=articulation_6d,
            articulation=rotation_6d_to_matrix(articulation_6d),
            betas=betas,
            log_depth=log_depth,
            depth=log_depth.exp(),
            existence_logits=existence,
            visibility_logits=visibility,
            joints_root_direct=joints_root,
            anchors_2d=anchors,
            anchors_2d_latent=anchors_latent,
            attention_heatmaps=joint_weights.reshape(
                batch,
                latent_frames,
                self.config.hand_queries,
                self.config.joints_per_hand,
                height,
                width,
            ),
        )
