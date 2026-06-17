"""KPConv U-Net with proper input embedding."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import KPConvBlock, KPConvNoResidual, KPConvLayer, _knn_neighbors


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """最远点采样（FPS），返回 (B, npoint) 索引。"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)
        d = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, d)
        farthest = distance.argmax(dim=-1)
    return centroids


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by indices."""
    B = points.shape[0]
    if idx.dim() == 2:
        S = idx.shape[1]
        return points[torch.arange(B, device=points.device).unsqueeze(-1).expand(B, S), idx]
    S, K = idx.shape[1], idx.shape[2]
    return points[torch.arange(B, device=points.device).unsqueeze(-1).unsqueeze(-1).expand(B, S, K), idx]


def _nearest_interpolate(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """最邻近插值上采样。"""
    B, C, N_src = features.shape
    N_tgt = idx.shape[1]
    f = features.permute(0, 2, 1).contiguous()
    batch_idx = torch.arange(B, device=features.device).view(B, 1).expand(B, N_tgt)
    gathered = f[batch_idx, idx]
    return gathered.permute(0, 2, 1).contiguous()


class EncoderBlock(nn.Module):
    """Encoder: 1-2 个 KPConvBlock + 步长 2 FPS 下采样。"""

    def __init__(self, in_feat: int, out_feat: int, n_blocks: int = 2) -> None:
        super().__init__()
        layers = [KPConvBlock(in_feat, out_feat)]
        for _ in range(n_blocks - 1):
            layers.append(KPConvBlock(out_feat, out_feat))
        self.blocks = nn.ModuleList(layers)

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor, neighbors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            features = block(xyz, features, neighbors)
        n_target = max(1, xyz.shape[1] // 2)
        fps_idx = _farthest_point_sample(xyz, n_target)
        new_xyz = _index_points(xyz, fps_idx)
        new_feat = _index_points(features.permute(0, 2, 1), fps_idx).permute(0, 2, 1)
        new_neighbors = _knn_neighbors(new_xyz, k=16)
        return new_xyz, new_feat, new_neighbors


class DecoderBlock(nn.Module):
    """Decoder: 上采样 + skip + KPConvBlock。"""

    def __init__(self, in_feat: int, skip_feat: int, out_feat: int, n_blocks: int = 2) -> None:
        super().__init__()
        self.skip_proj = nn.Conv1d(skip_feat, in_feat, kernel_size=1)
        layers = [KPConvNoResidual(2 * in_feat, out_feat)]
        for _ in range(n_blocks - 1):
            layers.append(KPConvNoResidual(out_feat, out_feat))
        self.blocks = nn.ModuleList(layers)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        skip_xyz: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N_tgt, _ = skip_xyz.shape
        _, _, N_skip = skip_features.shape
        d = torch.cdist(skip_xyz, xyz)
        nn_idx = d.argmin(dim=-1)
        up = _nearest_interpolate(features, nn_idx)
        if N_skip != N_tgt:
            skip_sampled = _nearest_interpolate(skip_features, nn_idx)
        else:
            skip_sampled = skip_features
        skip_proj = self.skip_proj(skip_sampled)
        concat = torch.cat([up, skip_proj], dim=1)
        neighbors = _knn_neighbors(skip_xyz, k=16)
        for block in self.blocks:
            concat = block(skip_xyz, concat, neighbors)
        return skip_xyz, concat


class KPConvUNet(nn.Module):
    """KPConv 5 层 encoder / 4 层 decoder U-Net with proper input embedding."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 9,
        K: int = 15,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.K = K
        feats = [32, 64, 128, 256, 512]

        # Input embedding: project raw features (RGB + intensity) to 32-dim
        # Also normalize coordinates (center + scale)
        self.xyz_bn = nn.BatchNorm1d(3)
        self.feat_embed = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )

        # Encoder
        self.enc1 = EncoderBlock(32, feats[0], n_blocks=2)
        self.enc2 = EncoderBlock(feats[0], feats[1], n_blocks=2)
        self.enc3 = EncoderBlock(feats[1], feats[2], n_blocks=2)
        self.enc4 = EncoderBlock(feats[2], feats[3], n_blocks=2)
        self.bottleneck = KPConvBlock(feats[3], feats[4])

        # Decoder
        self.dec4 = DecoderBlock(feats[4], feats[2], feats[3], n_blocks=2)
        self.dec3 = DecoderBlock(feats[3], feats[1], feats[2], n_blocks=2)
        self.dec2 = DecoderBlock(feats[2], feats[0], feats[1], n_blocks=2)
        self.dec1 = DecoderBlock(feats[1], feats[0], feats[0], n_blocks=2)

        # Final
        self.head = nn.Sequential(
            nn.Conv1d(feats[0], feats[0] // 2, kernel_size=1),
            nn.BatchNorm1d(feats[0] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(feats[0] // 2, num_classes, kernel_size=1),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            xyz: (B, N, 3) coordinates (already centered)
            features: (B, N, C_in) raw features (RGB + intensity)

        Returns:
            logits: (B, num_classes, N)
        """
        N_orig = xyz.shape[1]

        # Normalize coordinates
        xyz_norm = self.xyz_bn(xyz.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, 3)

        # Embed features
        f = self.feat_embed(features)  # (B, N, 32)
        f = f.permute(0, 2, 1)  # (B, 32, N)

        # Encoder
        neighbors = _knn_neighbors(xyz_norm, k=16)
        xyz1, f1, n1 = self.enc1(xyz_norm, f, neighbors)
        xyz2, f2, n2 = self.enc2(xyz1, f1, n1)
        xyz3, f3, n3 = self.enc3(xyz2, f2, n2)
        xyz4, f4, n4 = self.enc4(xyz3, f3, n3)
        fb = self.bottleneck(xyz4, f4, n4)

        # Decoder
        _, f4d = self.dec4(xyz4, fb, xyz3, f3)
        _, f3d = self.dec3(xyz3, f4d, xyz2, f2)
        _, f2d = self.dec2(xyz2, f3d, xyz1, f1)
        _, f1d = self.dec1(xyz1, f2d, xyz1, f1)

        # Interpolate back to original resolution
        if N_orig != xyz1.shape[1]:
            d = torch.cdist(xyz_norm, xyz1)
            nn_idx = d.argmin(dim=-1)
            f1d = _nearest_interpolate(f1d, nn_idx)

        # Classifier
        logits = self.head(f1d)
        return logits
