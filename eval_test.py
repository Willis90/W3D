"""Evaluate PointNet2 and visualize results."""
import torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from src.data.toronto3d_dataset import Toronto3DNPZDataset, CLASS_NAMES, NUM_CLASSES
from src.models.pointnet2 import PointNetPlusPlusSeg

device = "cuda"

# Colors per class (RGB 0-1)
CLASS_COLORS = [
    [0.5, 0.5, 0.5],   # Unclassified - gray
    [0.6, 0.8, 0.2],   # Ground - lime
    [1.0, 1.0, 0.0],   # Road_markings - yellow
    [0.0, 0.6, 0.0],   # Natural - green
    [1.0, 0.4, 0.0],   # Building - orange
    [1.0, 0.0, 1.0],   # Utility_line - magenta
    [0.0, 0.0, 1.0],   # Pole - blue
    [1.0, 0.0, 0.0],   # Car - red
    [0.8, 0.6, 0.0],   # Fence - brown
]

CLASS_COLORS_HEX = [
    '#808080', '#99CC33', '#FFFF00', '#009900',
    '#FF6600', '#FF00FF', '#0000FF', '#FF0000', '#CC9900',
]

# Load test data
test_ds = Toronto3DNPZDataset(
    npz_root="./Toronto_3D_npz",
    split="test",
    num_points=8192,
    augment=False,
)
print(f"Test: {len(test_ds)} crops from L004")

test_loader = DataLoader(test_ds, batch_size=4, num_workers=0, shuffle=False,
    collate_fn=lambda b: {"xyz": torch.stack([x[0] for x in b]),
                          "features": torch.stack([x[1] for x in b]),
                          "labels": torch.stack([x[2] for x in b])})

# Load model
model = PointNetPlusPlusSeg(in_channels=4, num_classes=NUM_CLASSES).to(device)
ckpt = torch.load("./checkpoints/best.pth", map_location="cpu")
if isinstance(ckpt, dict) and "model" in ckpt:
    ckpt = ckpt["model"]
model.load_state_dict(ckpt)
model.eval()
print(f"Model loaded")

# Evaluate - accumulate confusion matrix
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
all_preds, all_labels, all_coords = [], [], []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        xyz = batch["xyz"].to(device)
        feat = batch["features"].to(device)
        label = batch["labels"].to(device)
        logits = model(xyz, feat)
        pred = logits.argmax(dim=1)

        p = pred.flatten().cpu().numpy()
        l = label.flatten().cpu().numpy()
        c = xyz[:, :, :3].reshape(-1, 3).cpu().numpy()

        mask = l >= 0
        p, l, c = p[mask], l[mask], c[mask]
        all_preds.append(p)
        all_labels.append(l)
        all_coords.append(c)

        for ct in range(NUM_CLASSES):
            for cp in range(NUM_CLASSES):
                cm[ct, cp] += np.sum((l == ct) & (p == cp))

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_coords = np.concatenate(all_coords)
print(f"Total evaluated: {len(all_preds)} pts")

# 1. Per-class IoU bar chart
ious = []
for c in range(NUM_CLASSES):
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp
    iou = tp / (tp + fp + fn + 1e-8)
    ious.append(iou)
    print(f"  {CLASS_NAMES[c]:20s}: IoU={iou:.4f}")

miou = np.mean(ious)
oa = cm.trace() / cm.sum()
print(f"\nTest mIoU = {miou:.4f}  OA = {oa:.4f}")

# Save bar chart
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(NUM_CLASSES), ious, color=CLASS_COLORS_HEX, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels([n.replace('_', '\n') for n in CLASS_NAMES], fontsize=9)
ax.set_ylabel('IoU')
ax.set_title(f'Toronto-3D Per-Class IoU  (mIoU={miou:.4f}, OA={oa:.4f})')
ax.set_ylim(0, 0.7)
ax.axhline(miou, color='red', linestyle='--', linewidth=1, label=f'mIoU={miou:.4f}')
ax.legend()
for bar, iou in zip(bars, ious):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{iou:.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('/tmp/iou_bars.png', dpi=150)
print("Saved /tmp/iou_bars.png")

# 2. Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(10, 8))
cm_norm = cm.astype(float)
cm_norm = cm_norm / (cm_norm.sum(axis=1, keepdims=True) + 1e-8)
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels([n.replace('_', '\n') for n in CLASS_NAMES], fontsize=8, rotation=45, ha='right')
ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_xlabel('Predicted')
ax.set_ylabel('Ground Truth')
ax.set_title('Normalized Confusion Matrix')
plt.colorbar(im, ax=ax)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        v = cm_norm[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                color='white' if v > 0.5 else 'black', fontsize=7)
plt.tight_layout()
plt.savefig('/tmp/confusion_matrix.png', dpi=150)
print("Saved /tmp/confusion_matrix.png")

# 3. Point cloud color-coded visualization (2D projections)
# Sample points for visualization (too many otherwise)
np.random.seed(42)
n_vis = 50000
idx = np.random.choice(len(all_coords), min(n_vis, len(all_coords)), replace=False)
coords_vis = all_coords[idx]
preds_vis = all_preds[idx]
labels_vis = all_labels[idx]

pred_colors = np.array([CLASS_COLORS[p] for p in preds_vis])
gt_colors = np.array([CLASS_COLORS[l] for l in labels_vis])

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

def scatter_on_axis(ax, xy, colors, xlabel, ylabel, title):
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=0.1, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')

# Top view (X-Y) - GT vs Pred
scatter_on_axis(axes[0, 0], coords_vis[:, [0, 1]], gt_colors, 'X', 'Y', 'GT: Top View (X-Y)')
scatter_on_axis(axes[0, 1], coords_vis[:, [0, 1]], pred_colors, 'X', 'Y', 'Pred: Top View (X-Y)')
# Error map
error = (preds_vis != labels_vis).astype(float)
axes[0, 2].scatter(coords_vis[:, 0], coords_vis[:, 1], c=error, cmap='RdYlGn_r', s=0.1, alpha=0.3)
axes[0, 2].set_xlabel('X'); axes[0, 2].set_ylabel('Y')
axes[0, 2].set_title(f'Error Map (Acc={1-error.mean():.3f})')
axes[0, 2].set_aspect('equal')

# Side view (X-Z) - GT vs Pred
scatter_on_axis(axes[1, 0], coords_vis[:, [0, 2]], gt_colors, 'X', 'Z', 'GT: Side View (X-Z)')
scatter_on_axis(axes[1, 1], coords_vis[:, [0, 2]], pred_colors, 'X', 'Z', 'Pred: Side View (X-Z)')

# Legend
handles = [mpatches.Patch(color=c, label=n) for c, n in zip(CLASS_COLORS_HEX, CLASS_NAMES)]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Toronto-3D L003 Test Cloud  |  PointNet2  |  mIoU=%.4f  OA=%.4f' % (miou, oa),
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('/tmp/pointcloud_views.png', dpi=150, bbox_inches='tight')
print("Saved /tmp/pointcloud_views.png")

# 4. Per-cloud stats
print("\nDone! Images saved to /tmp/")
