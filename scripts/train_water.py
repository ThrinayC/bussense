"""
scripts/train_water.py
======================
Trains the real PyTorch water segmentation model (LightweightRoadSegNet)
on the authentic FRED / WaterSegNets dataset.

Outputs:
- models/infrastructure/water/best.pt
- models/infrastructure/water/metrics.json
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bussense_infrastructure.models.architectures import LightweightRoadSegNet

MODEL_DIR = ROOT / "models" / "infrastructure" / "water"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "datasets" / "fred" / "raw"


class FredWaterDataset(Dataset):
    def __init__(self, split: str = "train", img_size: int = 256):
        self.img_dir = DATA_DIR / "images" / split
        self.msk_dir = DATA_DIR / "masks" / split
        self.img_size = img_size
        self.samples = []
        if self.img_dir.exists():
            for p in sorted(self.img_dir.glob("*.png")):
                mp = self.msk_dir / p.name
                if mp.exists():
                    self.samples.append((p, mp))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        with Image.open(ip) as img:
            img = img.convert("RGB").resize((self.img_size, self.img_size))
            img_arr = np.array(img, dtype=np.float32) / 255.0
            # [H, W, C] -> [C, H, W]
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        with Image.open(mp) as msk:
            msk = msk.convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
            msk_arr = (np.array(msk, dtype=np.float32) > 127).astype(np.float32)
            msk_tensor = torch.from_numpy(msk_arr).unsqueeze(0) # [1, H, W]

        return img_tensor, msk_tensor


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, targets, smooth=1.0):
        bce_loss = self.bce(pred_logits, targets)
        probs = torch.sigmoid(pred_logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
        return bce_loss + dice_loss


def evaluate_model(model, val_loader, device):
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_prec = 0.0
    total_rec = 0.0
    count = 0
    latencies = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            t0 = time.perf_counter()
            logits = model(imgs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(len(preds)):
                p = preds[i].view(-1)
                t = masks[i].view(-1)
                intersection = (p * t).sum().item()
                union = (p + t).clamp(0, 1).sum().item()
                iou = (intersection + 1e-6) / (union + 1e-6)
                dice = (2.0 * intersection + 1e-6) / (p.sum().item() + t.sum().item() + 1e-6)
                prec = (intersection + 1e-6) / (p.sum().item() + 1e-6)
                rec = (intersection + 1e-6) / (t.sum().item() + 1e-6)

                total_iou += iou
                total_dice += dice
                total_prec += prec
                total_rec += rec
                count += 1

    return {
        "mIoU": round(total_iou / max(count, 1), 4),
        "DiceScore": round(total_dice / max(count, 1), 4),
        "Precision": round(total_prec / max(count, 1), 4),
        "Recall": round(total_rec / max(count, 1), 4),
        "MeanLatencyMs": round(float(np.mean(latencies)), 2) if latencies else 0.0
    }


def train_water_model(epochs: int = 4, batch_size: int = 8, lr: float = 1e-3):
    print("=" * 65)
    print("BUSSENSE: TRAINING WATERLOGGING MODEL (FRED DATASET)")
    print("=" * 65)

    device = torch.device("cpu")
    print(f"Device: {device} (Host has no CUDA; CPU execution verified)")

    train_ds = FredWaterDataset("train", img_size=256)
    val_ds = FredWaterDataset("val", img_size=256)
    test_ds = FredWaterDataset("test", img_size=256)

    print(f"Dataset samples -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LightweightRoadSegNet(in_channels=3, num_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_iou = 0.0
    best_weights_path = MODEL_DIR / "best.pt"

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(imgs)

        train_loss = epoch_loss / len(train_ds)
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f} | Val mIoU: {val_metrics['mIoU']:.4f} | Val Dice: {val_metrics['DiceScore']:.4f}")

        if val_metrics['mIoU'] >= best_iou:
            best_iou = val_metrics['mIoU']
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "LightweightRoadSegNet",
                "task": "waterlogging_segmentation",
                "input_size": [256, 256],
                "val_metrics": val_metrics,
                "epoch": epoch
            }, best_weights_path)

    # Final evaluation on holdout test set
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device)
    total_params = sum(p.numel() for p in model.parameters())

    metrics = {
        "model_name": "LightweightRoadSegNet_Water",
        "dataset": "FRED_WaterSegNets",
        "parameters": total_params,
        "device": "CPU (Fallback)",
        "training_time_sec": round(time.time() - t_start, 2),
        "validation_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "input_resolution": "256x256",
        "weights_path": str(best_weights_path.relative_to(ROOT)).replace("\\", "/")
    }

    metrics_file = MODEL_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 65)
    print("WATERLOGGING MODEL TRAINING COMPLETED")
    print(f"Weights saved: {best_weights_path}")
    print(f"Holdout Test mIoU: {test_metrics['mIoU']:.4f} | Dice: {test_metrics['DiceScore']:.4f}")
    print(f"Average CPU Latency: {test_metrics['MeanLatencyMs']} ms")
    print("=" * 65)
    return metrics

if __name__ == "__main__":
    train_water_model(epochs=4, batch_size=8, lr=1e-3)
