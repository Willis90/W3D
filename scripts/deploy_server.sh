#!/bin/bash
# W3D 服务器部署脚本
# 用法：
#   bash scripts/deploy_server.sh train         # 训练
#   bash scripts/deploy_server.sh eval <ckpt>   # 评估
#   bash scripts/deploy_server.sh infer <in> <out> <ckpt>  # 推理

set -euo pipefail

# 颜色
RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; BLUE=$'\e[34m'; RESET=$'\e[0m'
log()  { echo "${BLUE}[INFO]${RESET} $*"; }
warn() { echo "${YELLOW}[WARN]${RESET} $*" >&2; }
err()  { echo "${RED}[ERR ]${RESET} $*" >&2; }
ok()   { echo "${GREEN}[OK  ]${RESET} $*"; }

# 1. 激活 conda yolov8 环境
log "激活 conda 环境 yolov8"
if ! command -v conda &> /dev/null; then
    err "conda 未安装"
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yolov8 || { err "激活 yolov8 失败"; exit 1; }

# 2. 检查依赖
log "检查依赖"
python -c "import torch, open3d, sklearn, plyfile, tensorboard" \
    || { err "依赖缺失，请先运行: pip install open3d scikit-learn plyfile tensorboard easydict"; exit 1; }

# 3. 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    warn "未检测到 nvidia-smi，将以 CPU 跑（很慢）"
fi

# 4. 切到项目根
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
log "项目根: $PROJECT_ROOT"

MODE="${1:-train}"

case "$MODE" in
    train)
        log "开始训练"
        python src/train.py --config configs/default.yaml
        ;;
    eval)
        CKPT="${2:-checkpoints/best.pth}"
        log "开始评估: $CKPT"
        python src/eval.py --ckpt "$CKPT"
        ;;
    infer)
        IN_PLY="${2:?需要输入 ply 路径}"
        OUT_PLY="${3:?需要输出 ply 路径}"
        CKPT="${4:-checkpoints/best.pth}"
        log "开始推理: $IN_PLY -> $OUT_PLY (ckpt=$CKPT)"
        python src/inference.py --input "$IN_PLY" --output "$OUT_PLY" --ckpt "$CKPT"
        ;;
    *)
        err "未知模式: $MODE（train/eval/infer）"
        exit 2
        ;;
esac

ok "完成"
