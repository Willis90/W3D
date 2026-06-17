"""训练入口。"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.kpconv import KPConvUNet
from src.models.pointnet2 import PointNetPlusPlusSeg
from src.models.losses import WeightedCrossEntropyLoss


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_dataset(cfg: dict, split: str):
    """根据配置动态加载数据集。"""
    from src.data.dales_dataset import DALESDataset, compute_class_weights as dales_weights
    from src.data.toronto3d_dataset import Toronto3DNPZDataset, compute_class_weights as t3d_weights

    data_cfg = cfg["data"]
    dataset_name = data_cfg.get("dataset", "dales").lower()
    root = data_cfg["dataset_root"]

    if dataset_name == "toronto3d":
        print(f"[INFO] Loading Toronto3D dataset from {root}, split={split}")
        ds = Toronto3DNPZDataset(
            npz_root=root,
            split=split,
            num_points=data_cfg["num_points"],
            augment=data_cfg.get("augment", True) and split == "train",
        )
        weights = t3d_weights(ds)
        num_classes = 9
    else:
        print(f"[INFO] Loading DALES dataset from {root}, split={split}")
        ds = DALESDataset(
            root=root,
            split=split,
            num_points=data_cfg["num_points"],
            augment=data_cfg.get("augment", True) and split == "train",
        )
        weights = dales_weights(ds)
        num_classes = 8

    return ds, weights, num_classes


def _collate(batch):
    """将 (coords, features, labels) tuple 列表转为 dict。"""
    coords, features, labels = zip(*batch)
    return {
        "xyz": torch.stack(coords, dim=0),
        "features": torch.stack(features, dim=0),
        "labels": torch.stack(labels, dim=0),
    }


def _build_model(cfg: dict, num_classes: int):
    m = cfg["model"]
    model_type = m.get("type", "kpconv").lower()
    if model_type == "pointnet2" or model_type == "pointnet++":
        print(f"[INFO] Building PointNetPlusPlusSeg (in_channels={m.get('in_channels', 4)}, num_classes={num_classes})")
        return PointNetPlusPlusSeg(
            in_channels=m.get("in_channels", 4),
            num_classes=num_classes,
        )
    else:
        print(f"[INFO] Building KPConvUNet (in_channels={m.get('in_channels', 4)}, num_classes={num_classes})")
        return KPConvUNet(
            in_channels=m.get("in_channels", m.get("in_features", 4)),
            num_classes=num_classes,
            K=m["k"],
        )


def _build_optimizer(model: torch.nn.Module, cfg: dict):
    t = cfg["train"]
    name = t["optimizer"].lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=t["lr"],
            momentum=t["momentum"],
            weight_decay=t["weight_decay"],
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(opt, cfg: dict):
    t = cfg["train"]
    name = t["scheduler"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    raise ValueError(f"Unknown scheduler: {name}")


def _fast_mIoU(pred: torch.Tensor, label: torch.Tensor, num_classes: int) -> float:
    """快速计算 mIoU（不计 ignore）。"""
    pred = pred.flatten()
    label = label.flatten()
    mask = label >= 0
    pred = pred[mask]
    label = label[mask]
    ious = []
    for c in range(num_classes):
        p = pred == c
        l = label == c
        inter = (p & l).sum().item()
        union = (p | l).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def _validate(model, loader, device, num_classes) -> float:
    model.eval()
    miou_total = 0.0
    n = 0
    for batch in loader:
        xyz = batch["xyz"].to(device)
        feat = batch["features"].to(device)
        label = batch["labels"].to(device)
        logits = model(xyz, feat)
        pred = logits.argmax(dim=1)
        miou_total += _fast_mIoU(pred, label, num_classes)
        n += 1
    return miou_total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None, help="覆盖 dataset_root")
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="仅跑 N 步（快速测试）")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.data_root:
        cfg["data"]["dataset_root"] = args.data_root
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    device = args.device or cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[INFO] device = {device}")

    # 路径
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    train_ds, weights, num_classes = _build_dataset(cfg, "train")
    val_ds, _, _ = _build_dataset(cfg, "val")
    print(f"[INFO] train tiles = {len(train_ds)}, val tiles = {len(val_ds)}, num_classes = {num_classes}")
    print(f"[INFO] class weights = {weights.tolist()}")

    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg["num_workers"],
        drop_last=True,
        collate_fn=_collate,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        collate_fn=_collate,
    )

    # 模型 + 优化器 + 损失
    torch.manual_seed(cfg.get("seed", 42))
    model = _build_model(cfg, num_classes).to(device)
    opt = _build_optimizer(model, cfg)
    sched = _build_scheduler(opt, cfg)
    criterion = WeightedCrossEntropyLoss(weights, ignore_index=-100).to(device)

    # TensorBoard (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
    except Exception as e:
        print(f"[WARN] tensorboard unavailable: {e}; continuing without writer")
        writer = None

    best_miou = -1.0
    max_steps = args.max_steps
    for epoch in range(tcfg["epochs"]):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for step, batch in enumerate(train_loader):
            xyz = batch["xyz"].to(device)
            feat = batch["features"].to(device)
            label = batch["labels"].to(device)
            logits = model(xyz, feat)
            loss = criterion(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            if (step + 1) % tcfg["log_every_n_steps"] == 0:
                avg = loss_sum / (step + 1)
                lr = opt.param_groups[0]["lr"]
                print(f"  [ep {epoch} step {step+1}] loss={avg:.4f} lr={lr:.6f}")
                if writer is not None:
                    writer.add_scalar("train/loss", avg, epoch * len(train_loader) + step)
            if max_steps and (step + 1) >= max_steps:
                print(f"[TEST] 仅测试 {max_steps} 步，提前结束")
                break
        sched.step()

        if max_steps:
            print("[TEST] 1 epoch 完成，训练测试通过！")
            break

        # 验证
        if (epoch + 1) % tcfg["val_every_n_epochs"] == 0:
            miou = _validate(model, val_loader, device, num_classes)
            print(f"[ep {epoch}] val mIoU={miou:.4f}  time={time.time()-t0:.1f}s")
            if writer is not None:
                writer.add_scalar("val/mIoU", miou, epoch)
            if miou > best_miou:
                best_miou = miou
                torch.save({"model": model.state_dict(), "epoch": epoch, "miou": miou}, ckpt_dir / "best.pth")
                print(f"  [CKPT] best mIoU={miou:.4f} saved")
        if (epoch + 1) % 10 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pth")

    print(f"[DONE] best val mIoU = {best_miou:.4f}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()