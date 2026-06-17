"""KPConv 基础算子与 block。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fibonacci_sphere(n: int) -> torch.Tensor:
    """在单位球面上均匀采样 n 个点（核点位置）。"""
    indices = torch.arange(0, n, dtype=torch.float32) + 0.5
    phi = torch.acos(1.0 - 2.0 * indices / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * indices
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


def _knn_neighbors(query: torch.Tensor, k: int) -> torch.Tensor:
    """对 query (B, N, 3) 找 k 个最近邻，返回 (B, N, k) 索引。"""
    d = torch.cdist(query, query)
    _, idx = d.topk(k=k, largest=False)
    return idx


def _gather_features(features: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    """按邻居索引 gather。

    Args:
        features: (B, C, N)
        neighbors: (B, N, k)
    Returns:
        (B, C, N, k)
    """
    B, C, N = features.shape
    k = neighbors.shape[-1]
    f = features.permute(0, 2, 1)
    batch_idx = torch.arange(B, device=features.device).view(B, 1, 1).expand(B, N, k)
    gathered = f[batch_idx, neighbors]
    return gathered.permute(0, 3, 1, 2).contiguous()


class KPConvLayer(nn.Module):
    """Rigid KPConv 卷积层（核点位置固定，不学习）。"""

    def __init__(self, in_feat: int, out_feat: int, K: int = 15, sigma: float = 1.0) -> None:
        super().__init__()
        self.K = K
        self.sigma = sigma
        kernel_points = _fibonacci_sphere(K)
        self.register_buffer("kernel_points", kernel_points)
        self.mlp2 = nn.Sequential(
            nn.Linear(K * in_feat, out_feat),
            nn.LeakyReLU(0.1),
            nn.Linear(out_feat, out_feat),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz.shape
        k = neighbors.shape[-1]
        K = self.K
        nbr_xyz = _gather_features(xyz.permute(0, 2, 1), neighbors)
        nbr_feat = _gather_features(features, neighbors)
        rel = nbr_xyz - xyz.permute(0, 2, 1).unsqueeze(-1)
        kp = self.kernel_points.permute(1, 0).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        rel = rel.unsqueeze(-1)
        dist = torch.norm(rel - kp, dim=1)
        w = torch.clamp(1.0 - dist / self.sigma, min=0.0)
        w_sum = w.sum(dim=2, keepdim=True) + 1e-6
        w = w / w_sum
        nbr_feat_t = nbr_feat.permute(0, 2, 3, 1)
        aggregated = torch.einsum("bnki,bnkc->bnic", w, nbr_feat_t)
        agg_flat = aggregated.reshape(B, N, K * nbr_feat_t.shape[-1])
        out = self.mlp2(agg_flat)
        return out.permute(0, 2, 1)


class KPConvBlock(nn.Module):
    """KPConv + BN + LeakyReLU + 残差。"""

    def __init__(self, in_feat: int, out_feat: int, K: int = 15) -> None:
        super().__init__()
        self.kpconv = KPConvLayer(in_feat, out_feat, K=K)
        self.bn = nn.BatchNorm1d(out_feat)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        if in_feat != out_feat:
            self.shortcut: nn.Module = nn.Conv1d(in_feat, out_feat, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(features)
        out = self.kpconv(xyz, features, neighbors)
        out = self.bn(out)
        out = self.act(out + residual)
        return out


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
