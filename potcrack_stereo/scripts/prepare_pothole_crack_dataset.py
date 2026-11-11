#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepares a balanced 2-class (pothole, crack) segmentation dataset in YOLO format:
- Class 0: pothole (from PavementDamagesG7 and pothole600)
- Class 1: crack (from crack500 and cracks-and-potholes-in-road)
"""

import os
import shutil
from pathlib import Path
from typing import List
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "pothole_crack_yolo"

G7_DIR = PROJECT_ROOT / "data" / "PavementDamagesG7" / "prepared"
POTHOLE_MIX_DIR = Path(r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\data\pothole_mix\extracted\pothole-mix")
POTHOLE_MIX_YOLO = Path(r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\data\pothole_mix_yolo")


def mask_to_yolo_polygons(mask_path: Path, class_id: int = 1) -> List[str]:
    """Converts binary mask to normalized YOLO segmentation polygon lines."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    h, w = mask.shape[:2]
    # Threshold mask
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    lines = []
    for cnt in contours:
        # Ignore tiny specks (< 15 pixels area)
        if cv2.contourArea(cnt) < 15:
            continue
        # Approximate contour to reduce points
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
        # Normalize
        norm_pts = []
        for x, y in pts:
            norm_pts.append(f"{min(1.0, max(0.0, x / w)):.6f}")
            norm_pts.append(f"{min(1.0, max(0.0, y / h)):.6f}")
        lines.append(f"{class_id} " + " ".join(norm_pts))
    return lines


def build_dataset(max_train_per_class: int = 500, max_val_per_class: int = 100):
    print("Building 2-class (pothole, crack) segmentation dataset...")

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"pothole": {"train": 0, "val": 0, "test": 0}, "crack": {"train": 0, "val": 0, "test": 0}}

    # 1. Potholes from G7 (class 0)
    print("\nProcessing G7 Potholes...")
    for split in ["train", "val", "test"]:
        img_dir = G7_DIR / "images" / split
        lbl_dir = G7_DIR / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                # Copy image
                dst_img = OUTPUT_DIR / "images" / split / f"g7_{img_path.name}"
                shutil.copy2(img_path, dst_img)
                # Ensure class is 0
                dst_lbl = OUTPUT_DIR / "labels" / split / f"g7_{img_path.stem}.txt"
                new_lines = []
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) > 5:
                        parts[0] = "0"
                        new_lines.append(" ".join(parts))
                dst_lbl.write_text("\n".join(new_lines))
                counts["pothole"][split] += 1

    # 2. Potholes from pothole600 (class 0)
    print("Processing pothole600...")
    for split, limit in [("train", max_train_per_class), ("val", max_val_per_class)]:
        lbl_src = POTHOLE_MIX_YOLO / "labels" / split
        img_src = POTHOLE_MIX_YOLO / "images" / split
        if not lbl_src.exists():
            continue
        for lbl_path in lbl_src.glob("pothole600*.txt"):
            if counts["pothole"][split] >= limit:
                break
            if lbl_path.stat().st_size == 0:
                continue
            img_path = img_src / f"{lbl_path.stem}.jpg"
            if not img_path.exists():
                img_path = img_src / f"{lbl_path.stem}.png"
            if img_path.exists():
                shutil.copy2(img_path, OUTPUT_DIR / "images" / split / img_path.name)
                # Keep class 0
                shutil.copy2(lbl_path, OUTPUT_DIR / "labels" / split / lbl_path.name)
                counts["pothole"][split] += 1

    # 3. Cracks from crack500 & cracks-and-potholes-in-road (class 1)
    print("Processing Cracks (crack500, cracks-and-potholes-in-road)...")
    split_map = {"training": "train", "validation": "val", "testing": "test"}
    limit_map = {"train": max_train_per_class, "val": max_val_per_class, "test": 50}

    for raw_split, target_split in split_map.items():
        limit = limit_map[target_split]
        for ds_name in ["crack500", "cracks-and-potholes-in-road", "edmcrack600"]:
            img_dir = POTHOLE_MIX_DIR / raw_split / ds_name / "images"
            mask_dir = POTHOLE_MIX_DIR / raw_split / ds_name / "masks"
            if not img_dir.exists() or not mask_dir.exists():
                continue

            for img_path in img_dir.glob("*"):
                if counts["crack"][target_split] >= limit:
                    break
                mask_path = mask_dir / f"{img_path.stem}.png"
                if not mask_path.exists():
                    mask_path = mask_dir / f"{img_path.name}"
                if not mask_path.exists():
                    continue

                poly_lines = mask_to_yolo_polygons(mask_path, class_id=1)
                if poly_lines:
                    dst_img = OUTPUT_DIR / "images" / target_split / f"crack_{img_path.name}"
                    dst_lbl = OUTPUT_DIR / "labels" / target_split / f"crack_{img_path.stem}.txt"
                    shutil.copy2(img_path, dst_img)
                    dst_lbl.write_text("\n".join(poly_lines))
                    counts["crack"][target_split] += 1

    # Create dataset.yaml
    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: pothole
  1: crack
"""
    (OUTPUT_DIR / "dataset.yaml").write_text(yaml_content)
    print(f"\nDataset prepared at: {OUTPUT_DIR}")
    print(f"Counts: {counts}")
    print(f"YAML config saved to: {OUTPUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    build_dataset(max_train_per_class=350, max_val_per_class=75)
