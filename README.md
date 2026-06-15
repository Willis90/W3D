# W3D — 三维点云基础设施语义分割

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red)

从机载激光雷达（LiDAR）点云中自动分割出**地面、植被、建筑、电力线、车辆、卡车、围栏、桥梁**八类基础设施要素，为电力巡检、城市建模、智慧城市等下游应用提供像素级（点级）精度的语义标注。

## 特性

- **基于 KPConv（轻量版）**：核点卷积对电力线/围栏等细长几何结构友好
- **DALES 数据集**：8 类基础设施标注，机载激光标准基准
- **U-Net 编码器-解码器**：5 层 encoder / 4 层 decoder + skip connection
- **加权交叉熵**：自动按类别频次反比加权，缓和电力线等稀疏类别的不均衡
- **完整 pipeline**：训练 / 推理 / 评估 / 部署脚本齐全
- **CPU 也能跑**：不需要 GPU 也能 import + forward（仅推理慢）

## 目录结构

```
W3D/
├── docs/算法设计.md          # 算法设计文档
├── configs/default.yaml      # 训练配置
├── src/
│   ├── data/
│   │   ├── dales_dataset.py  # DALES 数据加载
│   │   ├── precompute_knn.py # KNN 预计算
│   │   └── augment.py        # 数据增强
│   ├── models/
│   │   ├── blocks.py         # KPConvLayer / Block
│   │   ├── kpconv.py         # KPConv U-Net
│   │   └── losses.py         # 加权交叉熵
│   ├── train.py              # 训练入口
│   ├── inference.py          # 推理入口
│   └── eval.py               # 评估入口
├── scripts/
│   ├── download_dales.sh     # 数据下载
│   ├── prepare_dales.py      # 数据切分
│   └── deploy_server.sh      # 部署脚本
├── tests/test_imports.py     # 单元测试
└── LICENSE
```

## 快速开始

### 1. 安装依赖

```bash
conda activate yolov8
pip install open3d scikit-learn plyfile tensorboard easydict
# KPConv 用的 KNN 加速（可选但推荐）
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

### 2. 数据准备

```bash
# 下载 DALES（需在 https://udayton.edu 注册）
bash scripts/download_dales.sh ./data/dales/raw

# 切分成 tile
python scripts/prepare_dales.py --root ./data/dales

# 预计算 KNN 索引
python src/data/precompute_knn.py --root ./data/dales --split train
python src/data/precompute_knn.py --root ./data/dales --split val
```

### 3. 训练

```bash
python src/train.py --config configs/default.yaml
# 或：bash scripts/deploy_server.sh train
```

### 4. 评估

```bash
python src/eval.py --ckpt checkpoints/best.pth
# 或：bash scripts/deploy_server.sh eval checkpoints/best.pth
```

### 5. 推理（输出带颜色的 ply）

```bash
python src/inference.py \
  --input ./data/dales/raw/11sta_0.ply \
  --output ./predictions/11sta_0_pred.ply \
  --ckpt checkpoints/best.pth
# 或：bash scripts/deploy_server.sh infer <in.ply> <out.ply> <ckpt>
```

## 服务器部署

服务器：RTX 5070 (12GB) / CUDA 12.8 / conda env `yolov8`

```bash
# 一键训练
bash scripts/deploy_server.sh train

# 一键评估
bash scripts/deploy_server.sh eval checkpoints/best.pth

# 一键推理
bash scripts/deploy_server.sh infer input.ply output.ply checkpoints/best.pth
```

## 单元测试

```bash
pytest tests/test_imports.py -v
```

## 引用 DALES 数据集

```
@inproceedings{varney2020dales,
  title={DALES: A Large-scale Aerial LiDAR Data Set for Semantic Segmentation},
  author={Varney, Nina and Asari, Vijayan K and Graehling, Quinn},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2020}
}
```

## License

MIT
