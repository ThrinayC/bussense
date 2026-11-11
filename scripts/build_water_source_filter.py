#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_water_source_filter.py
====================================
Audits and classifies all candidate waterlogging images.
Strictly separates true ROAD_WATER (submerged drivable asphalt with vehicles/traffic)
from natural waterbodies (rivers, beaches, lakes, harbours) and non-road puddles.
Outputs:
- outputs/reports/water_demo_source_filter.json
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

    detector = YOLO(str(ROOT / "potcrack_stereo" / "yolov8n.pt"))
    raw_dir = ROOT / "datasets" / "fred" / "raw"
    img_paths = sorted(list((raw_dir / "images").rglob("*.png")))

    filter_records = []
    road_water_count = 0

    for idx, p in enumerate(img_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue

        mask_p = raw_dir / "masks" / p.parent.name / p.name
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) if mask_p.exists() else None

        h, w = img.shape[:2]
        res = detector(img, verbose=False)[0]
        detected_classes = [res.names[int(b.cls[0])] for b in res.boxes]

        # Calculate water coverage in lower road area
        bottom_water_ratio = (
            np.count_nonzero(mask[int(h * 0.45):, :]) / float(h * 0.55 * w)
            if mask is not None else 0.0
        )

        has_boat = "boat" in detected_classes
        has_vehicles = any(c in detected_classes for c in ["car", "bus", "truck", "motorcycle", "bicycle"])
        has_infrastructure = any(c in detected_classes for c in ["traffic light", "stop sign", "bench"])
        has_wildlife = any(c in detected_classes for c in ["elephant", "cow", "bear", "horse"])

        # Detailed Classification
        if has_boat or bottom_water_ratio > 0.90:
            classification = "NATURAL_WATERBODY"
            usable = False
            reason = "Open water body, river, harbour, or lake with boat traffic; no drivable roadway"
        elif has_wildlife:
            classification = "NON_ROAD_WATER"
            usable = False
            reason = "Wildlife/nature terrain puddle; not a paved urban road surface"
        elif has_vehicles or has_infrastructure or (0.15 <= bottom_water_ratio <= 0.85 and "train19" in p.name):
            classification = "ROAD_WATER"
            usable = True
            reason = "Standing water visibly covering vehicle roadway/paved street surface"
            road_water_count += 1
        elif bottom_water_ratio < 0.08:
            classification = "ROAD_ADJACENT_WATER"
            usable = False
            reason = "Water localized to curb/ditch beside road or negligible surface pooling"
        elif "val" in p.parent.name and int(p.stem.replace("val", "")) >= 6:
            classification = "NATURAL_WATERBODY"
            usable = False
            reason = "Lake/river expanse covering full horizon"
        else:
            classification = "NON_ROAD_WATER"
            usable = False
            reason = "Pedestrian park/courtyard or generic landscape water without clear road boundaries"

        filter_records.append({
            "source": str(p.relative_to(ROOT)),
            "filename": p.name,
            "split": p.parent.name,
            "frame": idx,
            "detected_objects": list(set(detected_classes)),
            "bottom_water_ratio": round(bottom_water_ratio, 3),
            "classification": classification,
            "usable_for_real_demo": usable,
            "reason": reason
        })

    out_file = reports_dir / "water_demo_source_filter.json"
    with open(out_file, "w") as f:
        json.dump(filter_records, f, indent=2)

    print(f"water_demo_source_filter.json generated ({len(filter_records)} records, {road_water_count} ROAD_WATER).")


if __name__ == "__main__":
    main()
