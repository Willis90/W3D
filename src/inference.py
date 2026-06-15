"""推理入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from src.data.dales_dataset import DALES_COLORS, load_dales_ply
from src.models.kpconv import KPConvUNet


def _save_colored_ply(path: Path, xyz: np.ndarray, labels: np.ndarray) -> None:
    """把带标签的点云保存为带颜色的 ply。"""
    colors = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    for c, rgb in DALES_COLORS.items():
        mask = labels == c
        colors[mask] = rgb
    vertex = np.array(
        list(zip(xyz[:, 0], xyz[:, 1], xyz[:, 2], colors[:, 0], colors[:, 1], colors[:, 2], labels)),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "i4")],
    )
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(str(path))


def _sliding_window_tiles(xyz: np.ndarray, intensity: np.ndarray, num_points: int = 4096, stride: int = 2048):
    """把 scene 切成 (num_points) 个点的 tile，重叠区域用 stride 控制。"""
    n = xyz.shape[0]
    tiles = []
    for start in range(0, max(1, n - num_points + 1), stride):
        end = min(start + num_points, n)
        tile = {
            "xyz": xyz[start:end].copy(),
            "intensity": intensity[start:end].copy(),
            "start": start,
            "end": end,
        }
        # padding 如果不足 num_points
        if tile["xyz"].shape[0] < num_points:
            pad_n = num_points - tile["xyz"].shape[0]
            tile["xyz"] = np.concatenate([tile["xyz"], np.zeros((pad_n, 3), dtype=xyz.dtype)])
            tile["intensity"] = np.concatenate([tile["intensity"], np.zeros(pad_n, dtype=intensity.dtype)])
            tile["padded"] = True
        else:
            tile["padded"] = False
        tiles.append(tile)
    if n < num_points:
        # 整个 scene 不足一个 tile：padding
        tiles = [{
            "xyz": np.concatenate([xyz, np.zeros((num_points - n, 3), dtype=xyz.dtype)]),
            "intensity": np.concatenate([intensity, np.zeros(num_points - n, dtype=intensity.dtype)]),
            "start": 0,
            "end": n,
            "padded": True,
        }]
    return tiles


@torch.no_grad()
def predict_scene(model: KPConvUNet, xyz: np.ndarray, intensity: np.ndarray, device: str, num_points: int = 4096) -> np.ndarray:
    """滑窗预测一个 scene，返回 (N,) 标签。"""
    model.eval()
    tiles = _sliding_window_tiles(xyz, intensity, num_points=num_points, stride=num_points // 2)
    n = xyz.shape[0]
    prob_sum = np.zeros((n, model.num_classes), dtype=np.float32)
    prob_cnt = np.zeros(n, dtype=np.int32)
    for tile in tiles:
        s, e = tile["start"], tile["end"]
        # 真实点数
        real_n = e - s
        txyz = tile["xyz"]
        tint = tile["intensity"]
        feat = np.concatenate([txyz, tint[:, None]], axis=1).astype(np.float32)
        batch_xyz = torch.from_numpy(txyz[None]).float().to(device)
        batch_feat = torch.from_numpy(feat[None]).float().to(device)
        logits = model(batch_xyz, batch_feat)  # (1, C, num_points)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C, num_points)
        probs = probs.T  # (num_points, C)
        if not tile["padded"]:
            prob_sum[s:e] += probs
            prob_cnt[s:e] += 1
        else:
            prob_sum[s:s + real_n] += probs[:real_n]
            prob_cnt[s:s + real_n] += 1
    prob_cnt = np.maximum(prob_cnt, 1)
    avg = prob_sum / prob_cnt[:, None]
    return avg.argmax(axis=1).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="训练好的 .pth 路径")
    parser.add_argument("--input", required=True, help="输入 ply 路径")
    parser.add_argument("--output", required=True, help="输出 ply 路径")
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"[INFO] device = {args.device}")

    xyz, intensity, _ = load_dales_ply(args.input)
    print(f"[INFO] input points = {xyz.shape[0]}")

    model = KPConvUNet(in_features=4, num_classes=8, K=15)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model = model.to(args.device)

    labels = predict_scene(model, xyz, intensity, args.device, args.num_points)
    _save_colored_ply(Path(args.output), xyz, labels)
    print(f"[DONE] wrote {args.output}")


if __name__ == "__main__":
    main()
