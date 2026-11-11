#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep crack confidence thresholds across known crack and non-crack frames.
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
comb_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"))
data_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "left"

print("Model names:", comb_model.names)
assert comb_model.names[1] == "crack", f"Expected class 1 to be crack, got {comb_model.names}"

# Known crack frames: 62, 64, 65, 66, 68
crack_frames = [62, 64, 65, 66, 68]
# Known clean/non-crack frames: 10, 20, 30, 45
clean_frames = [10, 20, 30, 45]

thresholds = [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]

print("\n" + "="*70)
print("CRACK MODEL THRESHOLD SWEEP ON AUTHENTIC FOOTAGE")
print("="*70)

for th in thresholds:
    tp_cracks = 0
    fp_cracks = 0
    crack_details = []
    
    for f_idx in crack_frames:
        p = data_dir / f"frame_{f_idx:05d}.jpg"
        if not p.exists(): continue
        img = cv2.imread(str(p))
        res = comb_model(img, conf=th, verbose=False)[0]
        cnt = 0
        if res.boxes is not None:
            for b in res.boxes:
                if int(b.cls[0].item()) == 1:
                    cnt += 1
        tp_cracks += cnt
        crack_details.append(f"fr{f_idx}:{cnt}")
        
    for f_idx in clean_frames:
        p = data_dir / f"frame_{f_idx:05d}.jpg"
        if not p.exists(): continue
        img = cv2.imread(str(p))
        res = comb_model(img, conf=th, verbose=False)[0]
        cnt = 0
        if res.boxes is not None:
            for b in res.boxes:
                if int(b.cls[0].item()) == 1:
                    cnt += 1
        fp_cracks += cnt

    precision = tp_cracks / max(1, (tp_cracks + fp_cracks))
    recall_indicator = tp_cracks / len(crack_frames)
    f1 = 2 * (precision * (recall_indicator / 2.0)) / max(1e-4, precision + (recall_indicator / 2.0))
    
    print(f"Conf {th:.2f} | Crack Frames Total: {tp_cracks} ({' '.join(crack_details)}) | Clean Frames False Positives: {fp_cracks} | Precision: {precision:.2f}")

print("="*70)
