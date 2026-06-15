"""KPConv U-Net。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import KPConvBlock, _knn_neighbors
from .blocks import KPConvLayer  # noqa: F401  re-exported for downstream


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
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)  # (B, 1, 3)
        d = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        distance = torch.minimum(distance, d)
        farthest = distance.argmax(dim=-1)
    return centroids


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按 idx gather points。

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)
    Returns:
        (B, S, C) or (B, S, K, C)
    """
    B = points.shape[0]
    if idx.dim() == 2:
        S = idx.shape[1]
        return points[torch.arange(B, device=points.device).unsqueeze(-1).expand(B, S), idx]
    S, K = idx.shape[1], idx.shape[2]
    return points[torch.arange(B, device=points.device).unsqueeze(-1).unsqueeze(-1).expand(B, S, K), idx]


def _nearest_interpolate(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """最邻近插值上采样。

    Args:
        features: (B, C, N_src)
        idx: (B, N_tgt)  每个 target 点对应的 source 索引
    Returns:
        (B, C, N_tgt)
    """
    B, C, N_src = features.shape
    N_tgt = idx.shape[1]
    # features: (B, C, N_src) -> (B, N_src, C)
    f = features.permute(0, 2, 1).contiguous()
    # idx: (B, N_tgt) gather 沿 N_src 维
    batch_idx = torch.arange(B, device=features.device).view(B, 1).expand(B, N_tgt)
    gathered = f[batch_idx, idx]  # (B, N_tgt, C)
    return gathered.permute(0, 2, 1).contiguous()  # (B, C, N_tgt)


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
        # 下采样
        n_target = max(1, xyz.shape[1] // 2)
        fps_idx = _farthest_point_sample(xyz, n_target)
        new_xyz = _index_points(xyz, fps_idx)
        new_feat = _index_points(features.permute(0, 2, 1), fps_idx).permute(0, 2, 1)
        new_neighbors = _knn_neighbors(new_xyz, k=neighbors.shape[-1])
        return new_xyz, new_feat, new_neighbors


class DecoderBlock(nn.Module):
    """Decoder: 上采样 + skip + KPConvBlock（无残差，concat 自身即 skip connection）。

    维度约定：
    - in_feat:    encoder 输出（即 features 维度）
    - skip_feat:  对应的 encoder 跳跃连接的 skip 特征维度
    - 对齐逻辑：把 skip_features 1x1 conv 投影到 in_feat 维度，再 concat → 2*in_feat
    """

    def __init__(self, in_feat: int, skip_feat: int, out_feat: int, n_blocks: int = 2) -> None:
        super().__init__()
        # 1x1 conv 把 skip 特征投影到 in_feat 维度
        self.skip_proj = nn.Conv1d(skip_feat, in_feat, kernel_size=1)
        # concat 后维度 = in_feat + in_feat = 2*in_feat
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
        # 1. 最邻近插值上采样
        d = torch.cdist(skip_xyz, xyz)  # (B, N_tgt, N_src)
        nn_idx = d.argmin(dim=-1)  # (B, N_tgt)
        up = _nearest_interpolate(features, nn_idx)  # (B, in_feat, N_tgt)
        # 2. 投影 skip 特征到 in_feat 维度
        skip_proj = self.skip_proj(skip_features)  # (B, in_feat, N_tgt)
        # 3. 拼接
        concat = torch.cat([up, skip_proj], dim=1)  # (B, 2*in_feat, N_tgt)
        # 4. KPConv blocks
        neighbors = _knn_neighbors(skip_xyz, k=16)
        for block in self.blocks:
            concat = block(skip_xyz, concat, neighbors)
        return skip_xyz, concat


class KPConvNoResidual(nn.Module):
    """KPConv + BN + LeakyReLU（无残差，用于 decoder 块）。"""

    def __init__(self, in_feat: int, out_feat: int, K: int = 15) -> None:
        super().__init__()
        self.kpconv = KPConvLayer(in_feat, out_feat, K=K)
        self.bn = nn.BatchNorm1d(out_feat)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        out = self.kpconv(xyz, features, neighbors)
        out = self.bn(out)
        out = self.act(out)
        return out


class KPConvUNet(nn.Module):
    """KPConv 5 层 encoder / 4 层 decoder U-Net。"""

    def __init__(
        self,
        in_features: int = 4,
        num_classes: int = 8,
        K: int = 15,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.K = K
        feats = [32, 64, 128, 256, 512]
        # Encoder
        self.enc1 = EncoderBlock(in_features, feats[0], n_blocks=2)
        self.enc2 = EncoderBlock(feats[0], feats[1], n_blocks=2)
        self.enc3 = EncoderBlock(feats[1], feats[2], n_blocks=2)
        self.enc4 = EncoderBlock(feats[2], feats[3], n_blocks=2)
        self.bottleneck = KPConvBlock(feats[3], feats[4])
        # Decoder（dec_k 与 enc_k 对应：enc_k 输出的 skip 特征维度 = feats[k-1]）
        self.dec4 = DecoderBlock(feats[4], feats[2], feats[3], n_blocks=2)  # from 512, skip 128 -> 256
        self.dec3 = DecoderBlock(feats[3], feats[1], feats[2], n_blocks=2)  # from 256, skip 64 -> 128
        self.dec2 = DecoderBlock(feats[2], feats[0], feats[1], n_blocks=2)  # from 128, skip 32 -> 64
        self.dec1 = DecoderBlock(feats[1], in_features, feats[0], n_blocks=2)  # from 64, skip 4 -> 32
        # Classifier head
        self.head = nn.Conv1d(feats[0], num_classes, kernel_size=1)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            xyz: (B, N, 3)
            features: (B, N, C_in) — 注意是 (B, N, C)，不是 (B, C, N)
        Returns:
            logits: (B, num_classes, N)
        """
        # 转成 (B, C, N) 给 KPConv 用
        f = features.permute(0, 2, 1)  # (B, C, N)
        # 初始邻居
        neighbors = _knn_neighbors(xyz, k=16)
        # Encoder
        xyz1, f1, n1 = self.enc1(xyz, f, neighbors)
        xyz2, f2, n2 = self.enc2(xyz1, f1, n1)
        xyz3, f3, n3 = self.enc3(xyz2, f2, n2)
        xyz4, f4, n4 = self.enc4(xyz3, f3, n3)
        fb = self.bottleneck(xyz4, f4, n4)
        # Decoder
        _, f4d = self.dec4(xyz4, fb, xyz3, f3)
        _, f3d = self.dec3(xyz3, f4d, xyz2, f2)
        _, f2d = self.dec2(xyz2, f3d, xyz1, f1)
        _, f1d = self.dec1(xyz1, f2d, xyz, f)
        # Head
        logits = self.head(f1d)  # (B, num_classes, N)
        return logits
