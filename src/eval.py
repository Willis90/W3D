"""评估入口：计算 OA / mIoU / per-class IoU。"""
from __future__ import annotations

import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dales_dataset import DALESDataset, NUM_CLASSES
from src.data.dales_dataset import DALES_CLASSES
from src.models.kpconv import KPConvUNet


def _load_ckpt(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


@torch.no_grad()
def evaluate(model: KPConvUNet, loader: DataLoader, device: str) -> dict:
    model.eval()
    # 累积混淆矩阵
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for batch in loader:
        xyz = batch["xyz"].to(device)
        feat = batch["features"].to(device)
        label = batch["labels"].to(device)
        logits = model(xyz, feat)
        pred = logits.argmax(dim=1)
        p = pred.flatten().cpu().numpy()
        l = label.flatten().cpu().numpy()
        mask = l >= 0
        p = p[mask]
        l = l[mask]
        for c_true in range(NUM_CLASSES):
            for c_pred in range(NUM_CLASSES):
                cm[c_true, c_pred] += int(((l == c_true) & (p == c_pred)).sum())
    return _metrics_from_cm(cm)


def _metrics_from_cm(cm: np.ndarray) -> dict:
    inter = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    oa = inter.sum() / max(1, cm.sum())
    return {
        "OA": float(oa),
        "mIoU": float(iou.mean()),
        "per_class_IoU": {DALES_CLASSES[i + 1]: float(iou[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix": cm,
    }


def _print_table(metrics: dict) -> None:
    print("=" * 60)
    print(f"OA:   {metrics['OA']:.4f}")
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print("-" * 60)
    print(f"{'class':<15} {'IoU':>8}")
    print("-" * 60)
    for name, iou in metrics["per_class_IoU"].items():
        print(f"{name:<15} {iou:>8.4f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="./data/dales")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"[INFO] device = {args.device}")

    test_ds = DALESDataset(root=args.data_root, split="test", num_points=4096, augment=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = KPConvUNet(in_features=4, num_classes=NUM_CLASSES, K=15).to(args.device)
    _load_ckpt(model, args.ckpt)

    metrics = evaluate(model, loader, args.device)
    _print_table(metrics)


if __name__ == "__main__":
    main()
