#!/bin/bash
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov8

echo "=== 1. 环境检查 ==="
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import open3d; print('open3d:', open3d.__version__)"
pip show plyfile -q 2>/dev/null || pip install plyfile -q
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "=== 2. 模型前向测试 ==="
cd ~/projects/W3D
python -c "
import sys; sys.path.insert(0, 'src')
from models.kpconv import KPConvUNet
import torch
model = KPConvUNet(in_channels=4, num_classes=9, K=15).cuda()
print('Model params:', sum(p.numel() for p in model.parameters()))

# 模拟 Toronto3D: 4096 点, RGB+Intensity=4 channels
B, N, C = 2, 4096, 4
xyz = torch.randn(B, N, 3).cuda()
feat = torch.randn(B, N, C).cuda()
labels = torch.randint(0, 9, (B, N)).cuda()

logits = model(xyz, feat)
print('logits shape:', logits.shape, '(expected [2, 9, 4096])')
assert logits.shape == (B, 9, N), f'Shape mismatch: {logits.shape}'

loss = torch.nn.functional.cross_entropy(logits, labels)
print('Loss:', loss.item())
loss.backward()
print('Backward OK! GPU memory used:', torch.cuda.memory_allocated() // 1024**2, 'MB')
"

echo ""
echo "=== 3. Toronto3D 数据集加载测试 ==="
python -c "
import sys; sys.path.insert(0, 'src')
from data.toronto3d_dataset import Toronto3DDataset
ds = Toronto3DDataset(
    root='./Toronto_3D',
    split='train',
    num_points=4096,
    augment=False,
    subsample='grid',
    voxel_size=0.1
)
print('Dataset length:', len(ds))
coords, feat, labels = ds[0]
print('coords:', coords.shape, 'feat:', feat.shape, 'labels:', labels.shape)
print('Unique labels:', torch.unique(labels).tolist())
print('Dataset loaded OK!')
"

echo ""
echo "=== 4. 完整训练 1 epoch 测试 ==="
python src/train.py \
    --config configs/default.yaml \
    --data-root ./Toronto_3D \
    --epochs 1 \
    --device cuda 2>&1

echo ""
echo "=== 全部测试通过！ ==="