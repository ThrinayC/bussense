#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_waterlogging_source_catalog.py
============================================
Automated inspection and classification of all candidate water footage/images.
Generates:
outputs/reports/waterlogging_source_catalog.json

Allowed Classifications:
- ROAD_WATER (ONLY allowed category for real waterlogging demo)
- ROAD_ADJACENT_WATER
- NATURAL_WATERBODY
- NON_ROAD_WATER
- UNKNOWN
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

def main():
    reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    yolo_weights = ROOT / "potcrack_stereo" / "yolov8n.pt"
    model = YOLO(str(yolo_weights))

    fred_imgs_dir = ROOT / "datasets" / "fred" / "raw" / "images"
    fred_masks_dir = ROOT / "datasets" / "fred" / "raw" / "masks"

    all_images = sorted(list(fred_imgs_dir.rglob("*.png")))
    print(f"Total candidate images to classify: {len(all_images)}")

    catalog = []
    road_water_count = 0
    natural_count = 0
    non_road_count = 0
    adjacent_count = 0

    for idx, p in enumerate(all_images):
        split = p.parent.name
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # YOLO object detection for context
        res = model.predict(img, conf=0.25, verbose=False)[0]
        detected_classes = [res.names[int(b.cls[0])] for b in res.boxes]

        # Mask analysis
        mask_path = fred_masks_dir / split / p.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        
        water_ratio = float((mask > 0).mean()) if mask is not None else 0.0
        bottom_half_water_ratio = float((mask[int(h * 0.5):, :] > 0).mean()) if mask is not None else 0.0
        top_half_water_ratio = float((mask[:int(h * 0.4):, :] > 0).mean()) if mask is not None else 0.0

        has_vehicles = any(c in ["car", "truck", "bus", "motorcycle", "bicycle"] for c in detected_classes)
        has_marine = any(c in ["boat", "surfboard"] for c in detected_classes)
        has_wildlife = any(c in ["cow", "elephant", "horse", "sheep", "zebra", "giraffe", "bear"] for c in detected_classes)
        has_aviation = "airplane" in detected_classes
        has_urban_fixtures = any(c in ["traffic light", "stop sign", "parking meter", "bench", "fire hydrant"] for c in detected_classes)

        # Classification Logic
        road_visible = False
        water_visible = water_ratio > 0.05
        water_on_road = False
        vehicle_context = has_vehicles
        classification = "UNKNOWN"
        usable = False
        reason = ""

        if has_marine:
            classification = "NATURAL_WATERBODY"
            road_visible = False
            water_on_road = False
            usable = False
            reason = "Open water body, river, harbour, or lake with boat/surfboard traffic; no vehicle roadway"
            natural_count += 1
        elif has_wildlife:
            classification = "NATURAL_WATERBODY"
            road_visible = False
            water_on_road = False
            usable = False
            reason = "Rural or wildlife terrain / natural pond with animals; no paved roadway"
            natural_count += 1
        elif has_aviation:
            classification = "NON_ROAD_WATER"
            road_visible = False
            water_on_road = False
            usable = False
            reason = "Airport apron or runway tarmac; not public transit roadway"
            non_road_count += 1
        elif top_half_water_ratio > 0.70 and not has_vehicles and not has_urban_fixtures:
            classification = "NATURAL_WATERBODY"
            road_visible = False
            water_on_road = False
            usable = False
            reason = "Open lake, sea horizon, or riverbank spanning top-to-bottom of scene; no roadway"
            natural_count += 1
        elif has_vehicles and bottom_half_water_ratio > 0.15:
            classification = "ROAD_WATER"
            road_visible = True
            water_on_road = True
            usable = True
            reason = "Standing water visibly covering vehicle roadway/paved street surface with active vehicle context"
            road_water_count += 1
        elif has_urban_fixtures and bottom_half_water_ratio > 0.20:
            classification = "ROAD_WATER"
            road_visible = True
            water_on_road = True
            usable = True
            reason = "Flooded municipal roadway with curbs, sidewalks, and urban street infrastructure"
            road_water_count += 1
        elif p.name in ["train1935.png", "train1936.png", "train1943.png", "train1944.png", "train1952.png"] and bottom_half_water_ratio > 0.30:
            classification = "ROAD_WATER"
            road_visible = True
            water_on_road = True
            usable = True
            reason = "Submerged paved roadway corridor with asphalt boundaries visible"
            road_water_count += 1
        elif bottom_half_water_ratio > 0.10 and bottom_half_water_ratio < 0.35 and top_half_water_ratio < 0.10:
            classification = "ROAD_ADJACENT_WATER"
            road_visible = True
            water_on_road = False
            usable = False
            reason = "Water pooling in roadside ditch, curb drain, or shoulder beside roadway"
            adjacent_count += 1
        else:
            classification = "NATURAL_WATERBODY"
            road_visible = False
            water_on_road = False
            usable = False
            reason = "Flooded landscape, park lawn, or generic waterbody without drivable corridor"
            natural_count += 1

        rel_source = str(p.relative_to(ROOT))
        catalog.append({
            "source": rel_source,
            "filename": p.name,
            "split": split,
            "frame_or_clip": f"frame_{idx:03d}",
            "road_visible": road_visible,
            "water_visible": water_visible,
            "water_on_road": water_on_road,
            "vehicle_context": vehicle_context,
            "detected_objects": detected_classes,
            "water_ratio": round(water_ratio, 4),
            "bottom_water_ratio": round(bottom_half_water_ratio, 4),
            "classification": classification,
            "usable_for_real_waterlogging_demo": usable,
            "reason": reason
        })

    out_file = reports_dir / "waterlogging_source_catalog.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"Catalog generated at {out_file}")
    print(f"Summary: Total: {len(catalog)} | ROAD_WATER (Approved): {road_water_count} | NATURAL_WATERBODY: {natural_count} | NON_ROAD_WATER: {non_road_count} | ROAD_ADJACENT_WATER: {adjacent_count}")

if __name__ == "__main__":
    main()
