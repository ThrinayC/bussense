"""
train_pothole_seg.py - Fine-tune YOLO11-seg on G-7 Pothole Dataset

Trains and validates pothole segmentation on the session-split G-7 dataset.
Outputs:
- checkpoints/segmentation/best_pothole_seg.pt
- outputs/segmentation/metrics.json
- outputs/segmentation/sample_predictions/
"""

import os
import sys
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "data" / "PavementDamagesG7" / "prepared" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "segmentation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train_pothole_segmenter(epochs: int = 15, imgsz: int = 640, batch: int = 8):
    print("============================================================")
    print("TRAINING POTHOLE SEGMENTATION MODEL (YOLO11-seg)")
    print("============================================================")
    print(f"Dataset YAML : {DATASET_YAML}")
    print(f"Epochs       : {epochs}")
    print(f"Image Size   : {imgsz}")
    print(f"Batch Size   : {batch}")

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Prepared dataset config not found at {DATASET_YAML}")

    model = YOLO("yolo11n-seg.pt")

    # Train model
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=str(OUTPUT_DIR),
        name="g7_pothole_yolo11",
        exist_ok=True,
        workers=2,
        val=True,
        save=True,
        verbose=True
    )

    # Best weights path
    best_pt = OUTPUT_DIR / "g7_pothole_yolo11" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = OUTPUT_DIR / "g7_pothole_yolo11" / "weights" / "last.pt"

    dest_best_pt = CHECKPOINT_DIR / "best_pothole_seg.pt"
    if best_pt.exists():
        shutil.copyfile(best_pt, dest_best_pt)
        print(f"[Segmentation] Saved best checkpoint to: {dest_best_pt}")

    # Evaluate on Validation & Test Sets
    eval_model = YOLO(str(dest_best_pt))
    val_results = eval_model.val(data=str(DATASET_YAML), split="val", imgsz=imgsz, device=0)

    # Extract metrics
    metrics = {
        "model": "yolo11n-seg",
        "epochs": epochs,
        "imgsz": imgsz,
        "box_map50": float(val_results.box.map50) if hasattr(val_results, "box") else 0.0,
        "box_map50_95": float(val_results.box.map) if hasattr(val_results, "box") else 0.0,
        "mask_map50": float(val_results.seg.map50) if hasattr(val_results, "seg") else 0.0,
        "mask_map50_95": float(val_results.seg.map) if hasattr(val_results, "seg") else 0.0,
        "checkpoint_path": str(dest_best_pt)
    }

    metrics_file = OUTPUT_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Run inference on sample test images and save visualizations
    sample_out_dir = OUTPUT_DIR / "sample_predictions"
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    test_imgs_dir = PROJECT_ROOT / "data" / "PavementDamagesG7" / "prepared" / "images" / "test"
    test_files = list(test_imgs_dir.glob("*.jpg"))[:8]

    for tf in test_files:
        res = eval_model.predict(str(tf), conf=0.25, imgsz=imgsz, device=0)[0]
        annotated = res.plot()
        out_path = sample_out_dir / f"pred_{tf.name}"
        cv2.imwrite(str(out_path), annotated)

    print("\n------------------------------------------------------------")
    print("SEGMENTATION TRAINING COMPLETE:")
    print("------------------------------------------------------------")
    print(f"Mask mAP50    : {metrics['mask_map50']:.4f}")
    print(f"Mask mAP50-95 : {metrics['mask_map50_95']:.4f}")
    print(f"Box mAP50     : {metrics['box_map50']:.4f}")
    print(f"Metrics JSON  : {metrics_file}")
    print(f"Checkpoint    : {dest_best_pt}")
    print(f"Sample Visuals: {sample_out_dir}")
    print("============================================================\n")


if __name__ == "__main__":
    train_pothole_segmenter(epochs=15, imgsz=640, batch=8)
