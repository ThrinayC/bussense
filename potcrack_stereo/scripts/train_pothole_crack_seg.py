#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains and evaluates a unified 2-class (pothole, crack) segmentation model:
- Base: yolo11n-seg.pt
- Dataset: data/pothole_crack_yolo/dataset.yaml
- Evaluates on test set and verifies against protected pothole baseline.
"""

import sys
import json
import time
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "data" / "pothole_crack_yolo" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack"
PROTECTED_BASELINE = PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg_baseline_protected.pt"
REPORT_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "pothole_crack_seg_evaluation.json"


def train_and_evaluate():
    print("=" * 75)
    print(" TRAINING UNIFIED POTHOLE + CRACK SEGMENTATION MODEL (YOLO11n-seg)")
    print("=" * 75)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = YOLO("yolo11n-seg.pt")

    # Train for 15 epochs — fast convergence on transfer learning
    results = model.train(
        data=str(DATASET_YAML),
        epochs=15,
        imgsz=640,
        batch=16,
        device=device,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    best_ckpt = OUTPUT_DIR / "weights" / "best.pt"
    print(f"\nTraining complete! Best checkpoint: {best_ckpt}")

    # Evaluate on test set
    print("\n" + "=" * 75)
    print(" EVALUATION ON TEST DATASET")
    print("=" * 75)

    eval_model = YOLO(str(best_ckpt))
    val_res = eval_model.val(data=str(DATASET_YAML), split="test", imgsz=640, device=device)

    # Extract metrics
    # Mask metrics: mAP50, mAP50-95, precision, recall
    seg_map50 = float(val_res.seg.map50)
    seg_map = float(val_res.seg.map)
    seg_p = float(val_res.seg.mp)
    seg_r = float(val_res.seg.mr)

    # Class-specific mask metrics (0: pothole, 1: crack)
    p_map50 = float(val_res.seg.maps[0]) if len(val_res.seg.maps) > 0 else 0.0
    c_map50 = float(val_res.seg.maps[1]) if len(val_res.seg.maps) > 1 else 0.0

    print("\n[Unified Segmentation Test Results]")
    print(f"  Overall Mask mAP50:      {seg_map50:.3f}")
    print(f"  Overall Mask mAP50-95:   {seg_map:.3f}")
    print(f"  Overall Mask Precision:  {seg_p:.3f}")
    print(f"  Overall Mask Recall:     {seg_r:.3f}")
    print(f"  Class 0 (Pothole) mAP50: {p_map50:.3f}")
    print(f"  Class 1 (Crack) mAP50:   {c_map50:.3f}")

    # Baseline Pothole Validation on the same test split
    print("\nEvaluating protected pothole baseline on test split...")
    base_model = YOLO(str(PROTECTED_BASELINE))
    # Test pothole baseline (class 0 only)
    base_val = base_model.val(data=str(DATASET_YAML), split="test", imgsz=640, device=device)
    base_p_map50 = float(base_val.seg.map50)
    print(f"  Protected Baseline Mask mAP50: {base_p_map50:.3f}")

    # Regression check: pothole performance must not collapse
    pothole_preserved = (p_map50 >= base_p_map50 * 0.70) or (p_map50 > 0.40)
    print(f"\n  Pothole Branch Consistency Check: {'PASSED' if pothole_preserved else 'REGRESSION'}")

    report = {
        "status": "PASSED" if pothole_preserved else "FAILED",
        "checkpoint": str(best_ckpt),
        "overall_metrics": {
            "mask_map50": round(seg_map50, 4),
            "mask_map50_95": round(seg_map, 4),
            "mask_precision": round(seg_p, 4),
            "mask_recall": round(seg_r, 4)
        },
        "class_metrics": {
            "pothole": {
                "class_id": 0,
                "mask_map50": round(p_map50, 4),
                "baseline_map50": round(base_p_map50, 4),
                "delta": round(p_map50 - base_p_map50, 4)
            },
            "crack": {
                "class_id": 1,
                "mask_map50": round(c_map50, 4)
            }
        },
        "pothole_consistency_check": "PASSED" if pothole_preserved else "REGRESSION"
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {REPORT_PATH}\n")

    return report


if __name__ == "__main__":
    train_and_evaluate()
