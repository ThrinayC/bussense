"""
scripts/train_sign_condition.py
===============================
Trains the real PyTorch sign condition classifier (SignConditionNet)
on sign ROI crops extracted from authentic MTSVD/GTSDB images.

Outputs:
- models/infrastructure/sign_condition/best.pt
- models/infrastructure/sign_condition/metrics.json
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bussense_infrastructure.models.architectures import SignConditionNet

MODEL_DIR = ROOT / "models" / "infrastructure" / "sign_condition"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "datasets" / "mtsvd" / "raw"

CONDITIONS = ["NORMAL", "DAMAGED", "OCCLUDED"]
COND_TO_IDX = {c: i for i, c in enumerate(CONDITIONS)}


def apply_damage_augmentation(img: Image.Image) -> Image.Image:
    # Adds scratches, high-frequency noise and erosion/blur
    im = img.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    for _ in range(4):
        x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        draw.line([x0, y0, x1, y1], fill=(120, 120, 120), width=np.random.randint(1, 3))
    im = im.filter(ImageFilter.GaussianBlur(radius=0.8))
    return im


def apply_occlusion_augmentation(img: Image.Image) -> Image.Image:
    # Masks out a 30-50% rectangular block (foliage / obstacle)
    im = img.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    ox0 = np.random.randint(0, int(w * 0.4))
    oy0 = np.random.randint(0, int(h * 0.4))
    ox1 = min(w, ox0 + np.random.randint(int(w * 0.3), int(w * 0.6)))
    oy1 = min(h, oy0 + np.random.randint(int(h * 0.3), int(h * 0.6)))
    # Dark foliage-like color
    draw.rectangle([ox0, oy0, ox1, oy1], fill=(25, 45, 20))
    return im


class SignConditionDataset(Dataset):
    def __init__(self, split: str = "train", crop_size: int = 64):
        self.crop_size = crop_size
        self.samples = []

        split_dir = DATA_DIR / split
        coco_file = split_dir / "_annotations.coco.json"
        if not coco_file.exists():
            return

        with open(coco_file, "r") as f:
            coco = json.load(f)

        img_id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}

        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in img_id_to_file:
                fname = img_id_to_file[img_id]
                ip = split_dir / fname
                if ip.exists():
                    bbox = ann["bbox"] # [x, y, w, h]
                    if bbox[2] > 10 and bbox[3] > 10:
                        # Generate normal, damaged, and occluded variants
                        self.samples.append((ip, bbox, 0)) # NORMAL
                        self.samples.append((ip, bbox, 1)) # DAMAGED
                        self.samples.append((ip, bbox, 2)) # OCCLUDED

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, bbox, cond_idx = self.samples[idx]
        bx, by, bw, bh = bbox
        with Image.open(ip) as img:
            img = img.convert("RGB")
            crop = img.crop((int(bx), int(by), int(bx + bw), int(by + bh)))
            crop = crop.resize((self.crop_size, self.crop_size))

        if cond_idx == 1:
            crop = apply_damage_augmentation(crop)
        elif cond_idx == 2:
            crop = apply_occlusion_augmentation(crop)

        arr = np.array(crop, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        target = torch.tensor(cond_idx, dtype=torch.long)
        return tensor, target


def evaluate_condition(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    latencies = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            t0 = time.perf_counter()
            logits = model(imgs)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = round(correct / max(total, 1), 4)
    return {
        "Accuracy": acc,
        "MeanLatencyMs": round(float(np.mean(latencies)), 2) if latencies else 0.0
    }


def train_condition_model(epochs: int = 4, batch_size: int = 16, lr: float = 1e-3):
    print("=" * 65)
    print("BUSSENSE: TRAINING SIGN CONDITION CLASSIFIER")
    print("=" * 65)

    device = torch.device("cpu")
    train_ds = SignConditionDataset("train")
    val_ds = SignConditionDataset("val")
    test_ds = SignConditionDataset("test")

    print(f"Dataset crops -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SignConditionNet(in_channels=3, num_classes=3).to(device)
    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_weights_path = MODEL_DIR / "best.pt"

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = crit(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(imgs)

        train_loss = epoch_loss / max(len(train_ds), 1)
        val_metrics = evaluate_condition(model, val_loader, device)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f} | Val Acc: {val_metrics['Accuracy']:.4f}")

        if val_metrics['Accuracy'] >= best_acc:
            best_acc = val_metrics['Accuracy']
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "SignConditionNet",
                "conditions": CONDITIONS,
                "input_size": [64, 64],
                "val_metrics": val_metrics,
                "epoch": epoch
            }, best_weights_path)

    # Final evaluation on holdout test set
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_condition(model, test_loader, device)
    total_params = sum(p.numel() for p in model.parameters())

    metrics = {
        "model_name": "SignConditionNet",
        "dataset": "MTSVD_ROI_Crops",
        "conditions": CONDITIONS,
        "parameters": total_params,
        "device": "CPU (Fallback)",
        "training_time_sec": round(time.time() - t_start, 2),
        "validation_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "input_resolution": "64x64",
        "weights_path": str(best_weights_path.relative_to(ROOT)).replace("\\", "/")
    }

    metrics_file = MODEL_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 65)
    print("SIGN CONDITION CLASSIFIER TRAINING COMPLETED")
    print(f"Weights saved: {best_weights_path}")
    print(f"Holdout Test Accuracy: {test_metrics['Accuracy']:.4f}")
    print(f"Average CPU Latency: {test_metrics['MeanLatencyMs']} ms")
    print("=" * 65)
    return metrics

if __name__ == "__main__":
    train_condition_model(epochs=4, batch_size=16, lr=1e-3)
