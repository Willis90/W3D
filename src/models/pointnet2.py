"""PointNet++ style segmentation model with attention."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def fps_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by indices. points: (B, N, C). idx: (B, S) or (B, S, K).
    Returns: (B, S, C) or (B, S, K, C)"""
    B, N, C = points.shape
    device = points.device
    if idx.dim() == 2:
        S = idx.shape[1]
        idx_base = torch.arange(B, device=device).unsqueeze(1).expand(B, S)  # (B, S)
        return points[idx_base, idx]  # (B, S, C)
    else:
        S, K = idx.shape[1], idx.shape[2]
        idx_base = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2).expand(B, S, K)  # (B, S, K)
        return points[idx_base, idx]  # (B, S, K, C)


def square_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean distance."""
    return torch.sum((a.unsqueeze(2) - b.unsqueeze(1)) ** 2, dim=-1)


class PointNetSetAbstraction(nn.Module):
    """Set abstraction layer: FPS + ball query + PointNet layer."""

    def __init__(self, npoint: int, radius: float, k: int, in_channels: int, mlp: list[int], group_all: bool = False) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.k = k
        self.group_all = group_all

        self.mlp = nn.ModuleList()
        for out_ch in mlp:
            self.mlp.append(nn.Conv2d(in_channels, out_ch, 1))
            self.mlp.append(nn.BatchNorm2d(out_ch))
            self.mlp.append(nn.ReLU(inplace=True))
            in_channels = out_ch
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            xyz: (B, N, 3) coordinates
            feat: (B, C_in, N) features (xyz NOT included)

        Returns:
            new_xyz: (B, npoint, 3)
            new_points: (B, C', npoint)
        """
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            grouped_feat = feat.unsqueeze(3)  # (B, C, N, 1)
        else:
            fps_idx = fps_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)
            dists = square_distance(new_xyz, xyz)  # (B, npoint, N)
            knn_idx = dists.argsort(dim=-1)[:, :, :self.k]  # (B, npoint, k)
            # feat: (B, C, N) -> (B, N, C) for indexing
            feat_t = feat.permute(0, 2, 1)  # (B, N, C)
            grouped_feat_t = index_points(feat_t, knn_idx)  # (B, npoint, k, C)
            grouped_feat = grouped_feat_t.permute(0, 3, 1, 2)  # (B, C, npoint, k)

        new_points = self.mlp(grouped_feat)  # (B, C', npoint, k)
        new_points = new_points.max(dim=-1)[0]  # Max pooling: (B, C', npoint)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """Feature propagation: interpolate + concat + MLP."""

    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels1 + in_channels2, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Interpolate feat2 at xyz1 positions and concat with feat1."""
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape
        dists = square_distance(xyz1, xyz2)  # (B, N1, N2)
        dists, idx = dists.topk(k=min(3, N2), dim=-1, largest=False)  # (B, N1, 3)
        dists = 1.0 / (dists + 1e-8)
        norm = torch.sum(dists, dim=-1, keepdim=True)
        weight = dists / norm  # (B, N1, 3)
        interpolated = torch.sum(index_points(feat2.permute(0, 2, 1), idx) * weight.unsqueeze(-1), dim=-2)  # (B, N1, C2)

        new_feat = torch.cat([feat1.permute(0, 2, 1), interpolated], dim=-1)  # (B, N1, C1+C2)
        return self.mlp(new_feat.permute(0, 2, 1))  # (B, C_out, N1)


class PointNetPlusPlusSeg(nn.Module):
    """PointNet++ for semantic segmentation."""

    def __init__(self, in_channels: int = 4, num_classes: int = 9, use_xyz: bool = True) -> None:
        super().__init__()
        self.use_xyz = use_xyz
        # Encoder: 4 SA layers — xyz is only used for grouping (KNN/fps), not concat'd as channels
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, k=32, in_channels=in_channels, mlp=[32, 32, 64])
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, k=32, in_channels=64, mlp=[64, 64, 128])
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, k=32, in_channels=128, mlp=[128, 128, 256])
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=0.8, k=32, in_channels=256, mlp=[256, 256, 512])

        # Global feature
        self.global_softmax = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Feature propagation: 4 layers
        self.fp4 = PointNetFeaturePropagation(512, 256, 256)  # SA4(up,512)+SA3(skip,256)=768->256
        self.fp3 = PointNetFeaturePropagation(128, 256, 128)  # SA2(skip,128)+fp4(up,256)=384->128
        self.fp2 = PointNetFeaturePropagation(64, 128, 64)    # SA1(skip,64)+fp3(up,128)=192->64
        self.fp1 = PointNetFeaturePropagation(4, 64, 64)     # raw(skip,4)+fp2(up,64)=68->64

        # Final classifier: 324ch = fp1(64) + global(256) + raw(4)
        self.classifier = nn.Sequential(
            nn.Conv1d(324, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, num_classes, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            xyz: (B, N, 3)
            features: (B, N, C) or (B, C, N) — auto-detected and transposed to (B, C, N)

        Returns:
            logits: (B, num_classes, N)
        """
        # Training collate stacks as (B, N, C); SA layers need (B, C, N)
        if features.shape[1] == xyz.shape[1]:  # (B, N, C) -> transpose
            features = features.permute(0, 2, 1)

        # Encoder: SA layers use xyz only for grouping; features stay (B, C, N)
        l1_xyz, l1_feat = self.sa1(xyz, features)      # 1024 pts, 64 ch
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)    # 256 pts, 128 ch
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)    # 64 pts, 256 ch
        l4_xyz, l4_feat = self.sa4(l3_xyz, l3_feat)  # 16 pts, 512 ch

        # Global feature
        global_feat = self.global_softmax(l4_feat)  # (B, 256, 1)

        # Feature propagation: fpN(xyz_coarse, xyz_fine, skip_feat, up_feat)
        # up_feat is interp'd from coarser xyz to finer xyz, then concat with skip_feat
        l3_feat_up = self.fp4(l3_xyz, l4_xyz, l3_feat, l4_feat)  # 64pts: skip=SA3(256,64pts)+up=SA4(512,16pts)=768->256
        l2_feat_up = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat_up)  # 256pts: skip=SA2(128,256pts)+up=fp4(256,64pts)=384->128
        l1_feat_up = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_feat_up)  # 1024pts: skip=SA1(64,1024pts)+up=fp3(128,256pts)=192->64
        l0_feat_up = self.fp1(xyz, l1_xyz, features, l1_feat_up)   # Npts: skip=raw(4,Npts)+up=fp2(64,1024pts)=68->64

        # Concat global feat + raw features at N points for full context
        global_feat_expanded = global_feat.expand(-1, -1, l0_feat_up.shape[-1])  # (B, 256, N)
        # features is already (B, C, N) after the entry transpose
        x = torch.cat([l0_feat_up, global_feat_expanded, features], dim=1)  # (B, 324, N)

        # Classifier
        logits = self.classifier(x)  # (B, num_classes, N)
        return logits
