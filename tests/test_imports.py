"""Smoke test：所有模块能 import + 模型能 forward 一次。"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

# 让 import 找到 src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestImports(unittest.TestCase):
    def test_import_data(self):
        try:
            from src.data import augment, dales_dataset, precompute_knn  # noqa: F401
            from src.data.dales_dataset import DALESDataset, NUM_CLASSES, DALES_CLASSES  # noqa: F401
        except ModuleNotFoundError as e:
            # plyfile 在测试环境可能没装，但代码结构本身没问题
            if "plyfile" not in str(e):
                raise
            # 至少验证纯 Python 部分能 import
            from src.data import augment  # noqa: F401
            return
        self.assertEqual(NUM_CLASSES, 8)
        self.assertIn("ground", DALES_CLASSES.values())

    def test_import_models(self):
        from src.models import blocks, kpconv, losses  # noqa: F401
        from src.models.kpconv import KPConvUNet  # noqa: F401
        from src.models.losses import WeightedCrossEntropyLoss  # noqa: F401

    def test_import_train(self):
        import importlib
        for m in ["src.train", "src.inference", "src.eval"]:
            try:
                importlib.import_module(m)
            except ModuleNotFoundError as e:
                # plyfile 缺失时 import 会失败；这是环境问题不是代码问题
                if "plyfile" not in str(e):
                    raise


class TestKPConvForward(unittest.TestCase):
    def test_forward_shape(self):
        from src.models.kpconv import KPConvUNet
        model = KPConvUNet(in_features=4, num_classes=8, K=15)
        model.eval()
        B, N = 2, 256
        xyz = torch.randn(B, N, 3)
        feat = torch.randn(B, N, 4)
        with torch.no_grad():
            logits = model(xyz, feat)
        self.assertEqual(logits.shape, (B, 8, N))

    def test_loss_forward(self):
        from src.models.losses import WeightedCrossEntropyLoss
        weights = np.ones(8, dtype=np.float32)
        loss_fn = WeightedCrossEntropyLoss(weights, ignore_index=-100)
        logits = torch.randn(2, 8, 16)
        labels = torch.randint(0, 8, (2, 16))
        labels[0, 0] = -100
        loss = loss_fn(logits, labels)
        self.assertTrue(torch.isfinite(loss).item())


class TestAugment(unittest.TestCase):
    def test_rotation_preserves_shape(self):
        from src.data.augment import random_rotation_z, random_scale, random_jitter
        pts = np.random.randn(100, 4).astype(np.float32)
        for fn in [random_rotation_z, random_scale, random_jitter]:
            out = fn(pts)
            self.assertEqual(out.shape, pts.shape)


if __name__ == "__main__":
    unittest.main()
