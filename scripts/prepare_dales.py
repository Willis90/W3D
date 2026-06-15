"""把 DALES 原始 ply 切分成 tile，写入 data/dales/tiles/。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.dales_dataset import DALESDataset, DEFAULT_SPLIT, tile_scene, load_dales_ply


def prepare(root: str, num_points: int = 4096) -> None:
    root = Path(root)
    raw_dir = root / "raw"
    for split, names in DEFAULT_SPLIT.items():
        all_tiles = {"xyz": [], "intensity": [], "labels": []}
        for name in names:
            ply = raw_dir / f"{name}.ply"
            if not ply.exists():
                print(f"[WARN] missing {ply}, skipping")
                continue
            xyz, intensity, labels = load_dales_ply(ply)
            tiles = tile_scene(xyz, intensity, labels, num_points=num_points)
            for t in tiles:
                all_tiles["xyz"].append(t["xyz"])
                all_tiles["intensity"].append(t["intensity"])
                all_tiles["labels"].append(t["labels"])
        if not all_tiles["xyz"]:
            print(f"[WARN] no tiles for split={split}")
            continue
        out = root / f"{split}_tiles.npz"
        np.savez_compressed(
            out,
            xyz=np.stack(all_tiles["xyz"]),
            intensity=np.stack(all_tiles["intensity"]),
            labels=np.stack(all_tiles["labels"]),
        )
        print(f"[DONE] {split}: {len(all_tiles['xyz'])} tiles -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="DALES 数据根目录")
    parser.add_argument("--num-points", type=int, default=4096)
    args = parser.parse_args()
    prepare(args.root, args.num_points)


if __name__ == "__main__":
    main()
