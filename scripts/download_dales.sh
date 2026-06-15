#!/bin/bash
# DALES 数据下载脚本
# DALES 需在 https://udayton.edu/engineering/research/centers/vision_lab/research_and_data/w3d_data.php 注册后才能下载
# 用法：bash scripts/download_dales.sh <output_dir>

set -euo pipefail

OUT_DIR="${1:-./data/dales/raw}"
DALES_URL="https://udayton.edu/engineering/research/centers/vision_lab/research_and_data/w3d_data.php"

mkdir -p "$OUT_DIR"

echo "[INFO] 请到 $DALES_URL 注册并下载 DALES 数据集"
echo "[INFO] 下载后把 40 个 .ply 文件放到 $OUT_DIR"
echo "[INFO] 文件名格式：<scene>_<part>.ply (例如 11sta_0.ply)"
echo "[INFO] 当前期望的划分（见 src/data/dales_dataset.py DEFAULT_SPLIT）："
echo "  train: 28 scenes, val: 9 scenes, test: 6 scenes"
