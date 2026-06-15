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
    d = torch.cdist(query, query)  # (B, N, N)
    _, idx = d.topk(k=k, largest=False)  # (B, N, k)
    return idx


def _gather_features(features: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    """按邻居索引 gather。

    Args:
        features: (B, C, N)
        neighbors: (B, N, k)   每个值 ∈ [0, N)
    Returns:
        (B, C, N, k)
    """
    B, C, N = features.shape
    k = neighbors.shape[-1]
    # 关键：neighbors 索引是 per-batch 内的，不能把 batch 摊平后 gather
    # 用 batched gather：features[batch_idx, :, neighbors[b, n, k]]
    # features: (B, C, N) -> (B, N, C)
    f = features.permute(0, 2, 1)  # (B, N, C)
    batch_idx = torch.arange(B, device=features.device).view(B, 1, 1).expand(B, N, k)
    # gather
    gathered = f[batch_idx, neighbors]  # (B, N, k, C)
    return gathered.permute(0, 3, 1, 2).contiguous()  # (B, C, N, k)


class KPConvLayer(nn.Module):
    """Rigid KPConv 卷积层（核点位置固定，不学习）。"""

    def __init__(self, in_feat: int, out_feat: int, K: int = 15, sigma: float = 1.0) -> None:
        super().__init__()
        self.K = K
        self.sigma = sigma
        # 固定核点位置
        kernel_points = _fibonacci_sphere(K)
        self.register_buffer("kernel_points", kernel_points)
        # 把 (K * in_feat) 维特征映射到 out_feat
        self.mlp2 = nn.Sequential(
            nn.Linear(K * in_feat, out_feat),
            nn.LeakyReLU(0.1),
            nn.Linear(out_feat, out_feat),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """KPConv 前向。

        实现简化版：
        - 核点位置固定在球面 (K 个)
        - 对每个 query 点 x：先找 k 个邻居 {x_i, f_i}
        - 对每个核点 kp_j：在邻居中算"软分配"权重 w_ij = max(0, 1 - ||x - x_i - kp_j|| / σ)
        - 特征聚合：f'_j = Σ_i w_ij * f_i

        Args:
            xyz: (B, N, 3)
            features: (B, C_in, N)
            neighbors: (B, N, k)  — 每个 query 点的 k 个邻居索引
        Returns:
            (B, C_out, N)
        """
        B, N, _ = xyz.shape
        k = neighbors.shape[-1]
        K = self.K
        # 1. 收集邻居 xyz 和 features
        # xyz: (B, N, 3) -> permute to (B, 3, N) for gather
        nbr_xyz = _gather_features(xyz.permute(0, 2, 1), neighbors)  # (B, 3, N, k)
        nbr_feat = _gather_features(features, neighbors)  # (B, C_in, N, k)
        # 2. 相对位置（query 与邻居）
        rel = nbr_xyz - xyz.permute(0, 2, 1).unsqueeze(-1)  # (B, 3, N, k)
        # 3. 对每个核点 kp_j：在 (B, 3, N, k) rel 上加 kp_j，然后算 ||·||
        # kernel_points: (K, 3) -> (1, 3, 1, 1, K)
        kp = self.kernel_points.permute(1, 0).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, 3, 1, 1, K)
        rel = rel.unsqueeze(-1)  # (B, 3, N, k, 1)
        dist = torch.norm(rel - kp, dim=1)  # (B, N, k, K)
        w = torch.clamp(1.0 - dist / self.sigma, min=0.0)  # (B, N, k, K)
        w_sum = w.sum(dim=2, keepdim=True) + 1e-6  # (B, N, 1, K)
        w = w / w_sum  # 归一化
        # 4. 邻居特征 → (B, C_in, N, k) -> (B, N, k, C_in)
        nbr_feat_t = nbr_feat.permute(0, 2, 3, 1)  # (B, N, k, C_in)
        # 5. 用 w 对邻居特征做加权求和（每个核点一组权重）
        # w: (B, N, k, K) * nbr_feat_t: (B, N, k, C_in) -> 按 k 求和
        # 先 expand：w -> (B, N, k, K, 1), nbr_feat_t -> (B, N, k, 1, C_in)
        aggregated = torch.einsum("bnki,bnkc->bnic", w, nbr_feat_t)  # (B, N, K, C_in)
        # 6. (B, N, K, C_in) -> 拼成 (B, N, K*C_in) 过 MLP 到 out_feat
        agg_flat = aggregated.reshape(B, N, K * nbr_feat_t.shape[-1])
        # 用线性层映射到 out_feat
        out = self.mlp2(agg_flat)  # (B, N, C_out)
        return out.permute(0, 2, 1)  # (B, C_out, N)


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
