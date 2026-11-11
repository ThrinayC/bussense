#!/usr/bin/env python3
import sys
from pathlib import Path
from ultralytics import YOLO

pothole_model_path = "checkpoints/segmentation/best_pothole_seg.pt"
combined_model_path = "checkpoints/segmentation/combined_pothole_crack/weights/best.pt"

print("--- Pothole Model ---")
p_model = YOLO(pothole_model_path)
print("Names:", p_model.names)

print("\n--- Combined Model ---")
if Path(combined_model_path).exists():
    c_model = YOLO(combined_model_path)
    print("Names:", c_model.names)
else:
    print("Combined model NOT found at", combined_model_path)
