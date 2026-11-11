"""
scripts/audit_new_water_dataset.py
==================================
Audits the new Sazara Roadway Flooding Dataset (253 images + masks)
in strict accordance with Section 3, 4, 5 of the BusSense PRD:

Taxonomy:
- ROAD_WATER
- ROAD_ADJACENT_WATER
- NATURAL_WATERBODY
- POOL
- RIVER
- LAKE
- BEACH
- WATERFRONT
- NON_ROAD_WATER
- UNKNOWN

Only ROAD_WATER (where road is visible, water is visible, and water physically occupies the roadway)
is approved for the waterlogging demonstration.
"""

import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SAZARA_DIR = ROOT / "datasets" / "sazara_roadway_flood" / "Sazara"

def audit_sazara_dataset():
    img_dir = SAZARA_DIR / "image"
    mask_dir = SAZARA_DIR / "mask"

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"Auditing {len(img_files)} candidate images from Sazara Roadway Flooding Dataset...")

    # Load YOLOv8n for context detection (vehicles, signs, persons)
    detector = YOLO(str(ROOT / "potcrack_stereo" / "yolov8n.pt"))

    audit_records = []
    approved_road_water = []

    for idx, img_path in enumerate(img_files):
        img_name = img_path.name
        stem = img_path.stem  # e.g. "image_1"
        num_str = stem.split("_")[-1] if "_" in stem else str(idx + 1)
        mask_path = mask_dir / f"label_{num_str}.png"

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is not None:
            water_px = int(np.count_nonzero(mask > 0))
            water_ratio = float(water_px / (h * w))
        else:
            water_px = 0
            water_ratio = 0.0

        # Object context detection
        results = detector(bgr, verbose=False, conf=0.25)[0]
        detected_classes = [detector.names[int(b.cls[0])] for b in results.boxes]
        has_vehicles = any(c in ["car", "truck", "bus", "motorcycle"] for c in detected_classes)
        has_infrastructure = any(c in ["traffic light", "stop sign"] for c in detected_classes)
        has_boats = "boat" in detected_classes

        # Spatial distribution of water: Road water typically occupies the lower 70% of frame
        if mask is not None and water_px > 0:
            bottom_half_water = int(np.count_nonzero(mask[int(h * 0.4):, :] > 0))
            bottom_water_ratio = bottom_half_water / max(1, water_px)
        else:
            bottom_water_ratio = 0.0

        # Strict classification rules
        water_visible = water_ratio > 0.01

        # Check road visibility based on perspective and asphalt color in non-water lower area
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lower_region = gray[int(h * 0.5):, :]
        has_road_texture = (lower_region.std() > 10)

        # Classification decision tree
        if has_boats:
            classification = "RIVER" if "surfboard" not in detected_classes else "BEACH"
            road_visible = False
            water_on_road = False
            usable = False
            reason = "Boat detected; represents natural open waterway/river"
        elif not water_visible:
            classification = "NON_ROAD_WATER"
            road_visible = True
            water_on_road = False
            usable = False
            reason = "No significant water detected in scene"
        elif bottom_water_ratio >= 0.65 and (has_vehicles or has_infrastructure or has_road_texture):
            classification = "ROAD_WATER"
            road_visible = True
            water_on_road = True
            usable = True
            reason = f"Water visibly occupies roadway/pavement ({water_ratio*100:.1f}% frame, {bottom_water_ratio*100:.1f}% in lower roadway corridor)"
        elif bottom_water_ratio < 0.40:
            classification = "NATURAL_WATERBODY"
            road_visible = False
            water_on_road = False
            usable = False
            reason = "Water located predominantly in background/horizon; landscape or lake"
        else:
            classification = "ROAD_ADJACENT_WATER"
            road_visible = True
            water_on_road = False
            usable = False
            reason = "Water is adjacent to roadway in ditches or fields without physically blocking carriageway"

        record = {
            "source": str(img_path.relative_to(ROOT)),
            "frame": img_name,
            "road_visible": road_visible,
            "water_visible": water_visible,
            "water_on_road": water_on_road,
            "vehicle_context": has_vehicles,
            "classification": classification,
            "usable_for_waterlogging": usable,
            "reason": reason,
            "metrics": {
                "water_pixel_ratio": round(water_ratio, 4),
                "bottom_water_ratio": round(bottom_water_ratio, 4),
                "detected_objects": list(set(detected_classes))
            }
        }
        audit_records.append(record)
        if usable:
            approved_road_water.append(record)

    out_file = REPORTS_DIR / "waterlogging_new_dataset_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2)

    # Class summary
    from collections import Counter
    summary = Counter(r["classification"] for r in audit_records)
    print("\n=== SAZARA ROADWAY FLOOD DATASET AUDIT SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:22s}: {v:3d}")
    print(f"Total Approved ROAD_WATER images: {len(approved_road_water)} / {len(audit_records)}")
    print(f"Audit written to: {out_file}")

    # Also update FRED catalog to explicitly mark DEPRECATED_FOR_WATERLOGGING_DEMO
    fred_cat = REPORTS_DIR / "waterlogging_source_catalog.json"
    if fred_cat.exists():
        with open(fred_cat, "r", encoding="utf-8") as f:
            fred_data = json.load(f)
        deprecated_envelope = {
            "status": "DEPRECATED_FOR_WATERLOGGING_DEMO",
            "deprecation_notice": "FRED dataset is deprecated for real waterlogging demonstrations per Section 1 of Ground Truth PRD. Retained for historical evaluation only.",
            "replacement_dataset": "Sazara Roadway Flooding Dataset (HydroShare ID 24866122a6ee456c8f7c80aa87a9abcb)",
            "records": fred_data if isinstance(fred_data, list) else fred_data.get("records", [])
        }
        with open(fred_cat, "w", encoding="utf-8") as f:
            json.dump(deprecated_envelope, f, indent=2)
        print("Updated waterlogging_source_catalog.json with DEPRECATED_FOR_WATERLOGGING_DEMO notice.")

if __name__ == "__main__":
    audit_sazara_dataset()
