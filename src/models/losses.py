"""加权交叉熵。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """根据类别频次自动算权重的交叉熵。"""

    def __init__(self, class_weights: np.ndarray | torch.Tensor, ignore_index: int = -100) -> None:
        super().__init__()
        if isinstance(class_weights, np.ndarray):
            class_weights = torch.from_numpy(class_weights).float()
        self.register_buffer("weight", class_weights)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: (B, C, N)  labels: (B, N)。"""
        return F.cross_entropy(logits, labels, weight=self.weight, ignore_index=self.ignore_index)
