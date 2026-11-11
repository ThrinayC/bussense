"""
prepare_g7.py - PavementDamagesG-7 Dataset Preparation & Session-Based Split

Performs:
1. Extraction and pairing of left (_i.jpg) and right (_d.jpg) stereo images.
2. Extraction of binary pothole masks from ANOTATION using official RGB (42, 125, 209).
3. Session-based grouping to prevent temporal leakage across train/val/test splits.
4. Export of binary masks and YOLOv11 polygon labels for segmentation training.
5. Generation of dataset.yaml and split metadata summary.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
G7_ROOT = PROJECT_ROOT / "data" / "PavementDamagesG7" / "extracted"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "PavementDamagesG7" / "prepared"

POTHOLE_BGR = np.array([209, 125, 42], dtype=np.int16) # RGB (42, 125, 209)
COLOR_TOLERANCE = 18
MIN_POTHOLE_PIXELS = 80


def extract_pothole_mask(ann_img: np.ndarray) -> np.ndarray:
    """
    Extract binary uint8 mask (0 or 255) for pothole class.
    ann_img is BGR.
    """
    diff = np.abs(ann_img.astype(np.int16) - POTHOLE_BGR)
    mask = (diff[:, :, 0] <= COLOR_TOLERANCE) & \
           (diff[:, :, 1] <= COLOR_TOLERANCE) & \
           (diff[:, :, 2] <= COLOR_TOLERANCE)
    
    binary_mask = np.zeros(mask.shape, dtype=np.uint8)
    binary_mask[mask] = 255

    # Morphological cleanup for speckle noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return binary_mask


def mask_to_yolo_polygons(mask: np.ndarray) -> List[List[float]]:
    """
    Converts binary mask (H, W) to normalized YOLO polygon coordinates:
    class_id x1 y1 x2 y2 ... xn yn
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_POTHOLE_PIXELS:
            continue
        # Approximate contour to reduce vertices
        epsilon = 0.004 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        
        pts = approx.reshape(-1, 2).astype(np.float32)
        # Normalize to [0, 1]
        norm_pts = []
        for x, y in pts:
            norm_pts.append(float(np.clip(x / w, 0.0, 1.0)))
            norm_pts.append(float(np.clip(y / h, 0.0, 1.0)))
        polygons.append(norm_pts)

    return polygons


def get_session_id(stem: str) -> str:
    """
    Groups stems into contiguous recording sessions.
    Examples:
      '2024-07-30_16-09-32-512' -> '2024-07-30_16'
      '2024-08-26_15-35-12-001' -> '2024-08-26_15'
      'img_15-48-46-778' -> 'img_15-48'
    """
    if stem.startswith("2024") or stem.startswith("2025"):
        parts = stem.split("_")
        date = parts[0]
        hour = parts[1].split("-")[0] if len(parts) > 1 else "00"
        return f"{date}_{hour}"
    elif stem.startswith("img_"):
        parts = stem.split("-")
        return f"{parts[0]}-{parts[1]}"
    else:
        return stem[:10]


def prepare_dataset(split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)):
    img_dir = G7_ROOT / "IMAGES"
    ann_dir = G7_ROOT / "ANOTATION"

    if not img_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(f"G-7 dataset not found at {G7_ROOT}")

    print("============================================================")
    print("G-7 PAVEMENT DAMAGES: DATASET PREPARATION & SPLIT")
    print("============================================================")

    # 1. Discover matching stereo pairs
    all_files = os.listdir(img_dir)
    stems_left = {f[:-6] for f in all_files if f.endswith("_i.jpg")}
    stems_right = {f[:-6] for f in all_files if f.endswith("_d.jpg")}
    stereo_stems = sorted(list(stems_left & stems_right))

    print(f"Total Left images: {len(stems_left)}")
    print(f"Total Right images: {len(stems_right)}")
    print(f"Total Stereo pairs: {len(stereo_stems)}")

    # 2. Inspect annotations and filter frames with pothole instances
    pothole_records = []
    session_map: Dict[str, List[dict]] = {}

    for stem in stereo_stems:
        ann_path = ann_dir / f"{stem}_i.png"
        left_img_path = img_dir / f"{stem}_i.jpg"
        right_img_path = img_dir / f"{stem}_d.jpg"

        if not ann_path.exists():
            continue

        ann_img = cv2.imread(str(ann_path))
        if ann_img is None:
            continue

        mask = extract_pothole_mask(ann_img)
        pothole_pixel_count = int(np.sum(mask > 0))

        session_id = get_session_id(stem)
        record = {
            "stem": stem,
            "session_id": session_id,
            "left_img": left_img_path,
            "right_img": right_img_path,
            "ann_path": ann_path,
            "pothole_pixel_count": pothole_pixel_count,
            "has_pothole": pothole_pixel_count >= MIN_POTHOLE_PIXELS
        }

        if session_id not in session_map:
            session_map[session_id] = []
        session_map[session_id].append(record)

        if record["has_pothole"]:
            pothole_records.append(record)

    total_sessions = len(session_map)
    sessions_with_potholes = sorted(list({r["session_id"] for r in pothole_records}))
    print(f"Total sessions: {total_sessions}")
    print(f"Sessions containing potholes: {len(sessions_with_potholes)}")
    print(f"Total frames with valid pothole annotations: {len(pothole_records)}")

    # 3. Session-based split (No temporal frame leakage!)
    # Deterministic balanced partition of pothole sessions and background sessions:
    train_sessions = {
        "img_16-10", "img_16-06", "2024-08-26_15", "img_15-56", "img_16-03", "img_16-23",
        # Background sessions for train (~70% of background)
        "2024-09-30_15", "img_15-58", "img_15-30", "img_15-57", "img_16-01", "img_15-49", "img_15-15"
    }
    val_sessions = {
        "2024-09-16_15", "img_16-22",
        # Background sessions for val (~15% of background)
        "2024-07-30_16", "img_15-40", "img_16-04", "img_15-16", "img_16-19"
    }
    test_sessions = {
        "img_15-48", "img_16-21",
        # Background sessions for test (~15% of background)
        "2024-09-30_16", "2024-08-26_16", "img_15-34", "img_15-31", "img_16-58"
    }

    # 4. Prepare directory layout
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "images_right" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "masks" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 5. Process and copy files
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_pothole_counts = {"train": 0, "val": 0, "test": 0}

    for sess, records in session_map.items():
        if sess in train_sessions:
            split = "train"
        elif sess in val_sessions:
            split = "val"
        else:
            split = "test"

        for rec in records:
            stem = rec["stem"]
            # Copy Left RGB
            dest_left = OUTPUT_ROOT / "images" / split / f"{stem}.jpg"
            shutil.copyfile(rec["left_img"], dest_left)

            # Copy Right RGB (for stereo evaluation/testing)
            dest_right = OUTPUT_ROOT / "images_right" / split / f"{stem}.jpg"
            shutil.copyfile(rec["right_img"], dest_right)

            # Process mask
            ann_img = cv2.imread(str(rec["ann_path"]))
            mask = extract_pothole_mask(ann_img) if ann_img is not None else np.zeros((1080, 1920), dtype=np.uint8)
            dest_mask = OUTPUT_ROOT / "masks" / split / f"{stem}.png"
            cv2.imwrite(str(dest_mask), mask)

            # Generate YOLO polygons
            polys = mask_to_yolo_polygons(mask)
            label_file = OUTPUT_ROOT / "labels" / split / f"{stem}.txt"
            with open(label_file, "w", encoding="utf-8") as lf:
                for poly in polys:
                    line = "0 " + " ".join(f"{coord:.6f}" for coord in poly) + "\n"
                    lf.write(line)

            split_counts[split] += 1
            if rec["has_pothole"]:
                split_pothole_counts[split] += 1

    # 6. Create dataset.yaml for YOLO training
    yaml_content = f"""# G-7 Pothole Segmentation Dataset Config
path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: pothole
"""
    with open(OUTPUT_ROOT / "dataset.yaml", "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    metadata = {
        "dataset_name": "PavementDamagesG-7",
        "pothole_rgb": [42, 125, 209],
        "split_summary": {
            split: {
                "total_frames": split_counts[split],
                "pothole_frames": split_pothole_counts[split],
                "sessions": sorted(list(
                    train_sessions if split == "train" else (val_sessions if split == "val" else test_sessions)
                ))
            }
            for split in ["train", "val", "test"]
        }
    }
    with open(OUTPUT_ROOT / "g7_metadata.json", "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=4)

    print("\n------------------------------------------------------------")
    print("SPLIT SUMMARY:")
    print("------------------------------------------------------------")
    for s in ["train", "val", "test"]:
        print(f"  {s.upper():<6}: {split_counts[s]} total frames | {split_pothole_counts[s]} pothole frames | Sessions: {len(metadata['split_summary'][s]['sessions'])}")
    print(f"\nPrepared dataset exported to: {OUTPUT_ROOT}")
    print(f"YOLO configuration saved to: {OUTPUT_ROOT / 'dataset.yaml'}")
    print("============================================================\n")


if __name__ == "__main__":
    prepare_dataset()
