#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
combined_model_path = PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"

model = YOLO(str(combined_model_path))
print("Loaded model classes:", model.names)

data_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "left"
left_files = sorted(list(data_dir.glob("*.jpg")))

print(f"\nTesting on {len(left_files)} frames in {data_dir.name}...")

test_indices = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]

for idx in test_indices:
    if idx >= len(left_files):
        continue
    img = cv2.imread(str(left_files[idx]))
    H, W = img.shape[:2]
    
    # Test at conf 0.10, 0.15, 0.20
    for conf in [0.10, 0.15, 0.20]:
        res = model(img, conf=conf, verbose=False)[0]
        n_boxes = len(res.boxes) if res.boxes is not None else 0
        pothole_cnt = 0
        crack_cnt = 0
        if n_boxes > 0:
            for b in res.boxes:
                c_id = int(b.cls[0].item())
                if c_id == 0:
                    pothole_cnt += 1
                elif c_id == 1:
                    crack_cnt += 1
        print(f"Frame {idx:03d} (conf={conf:.2f}): {n_boxes} total -> {pothole_cnt} potholes, {crack_cnt} cracks")
