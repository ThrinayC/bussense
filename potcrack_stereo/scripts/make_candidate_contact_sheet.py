#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Phase 11: Candidate Video Selection & Contact Sheet
Generates outputs/demo/vehicle_pothole_candidate_contact_sheet.png
Visualizes candidate stereo sequences containing BOTH vehicles and potholes:
- Candidate 1: main_street_sbs.mp4 (23.1s, 1280x360@60fps SBS)
- Candidate 2: driving_full_sbs_16s.mp4 (15.9s, 1920x540@60fps SBS)
- Candidate 3: gitc_speed_hump.mp4 (54.3s, 640x336@30fps SBS)
- Candidate 4: PavementDamagesG7 / ZED 2i Stereo (1920x1080@30fps Left+Right)
"""

import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(r"C:\Users\Integra\.gemini\antigravity\brain\bdb57915-d8a4-47a7-b023-aa7ea297fbb4")

pothole_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg.pt"))
vehicle_model = YOLO("yolov8n.pt")


def get_candidate_frame(src_path: Path, is_video: bool = True, frame_idx: int = 150):
    """Extract Left and Right RGB images from source."""
    if is_video:
        cap = cv2.VideoCapture(str(src_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None, None
        w = frame.shape[1]
        half_w = w // 2
        left = frame[:, :half_w]
        right = frame[:, half_w:]
        return left, right
    else:
        # Image pair
        left = cv2.imread(str(src_path))
        right_path = str(src_path).replace("_i.jpg", "_d.jpg")
        right = cv2.imread(right_path) if os.path.exists(right_path) else left
        return left, right


def annotate_frame(left: np.ndarray, title: str, meta_str: str) -> np.ndarray:
    """Run detectors and draw informative overlay."""
    vis = left.copy()
    h, w = vis.shape[:2]

    # 1. Pothole segmentations
    p_res = pothole_model(left, conf=0.25, verbose=False)
    p_count = 0
    if len(p_res[0].boxes) > 0:
        if p_res[0].masks is not None:
            for mask_poly in p_res[0].masks.xy:
                pts = np.array(mask_poly, dtype=np.int32)
                if len(pts) > 2:
                    cv2.fillPoly(vis, [pts], (255, 128, 0))  # Cyan fill in BGR
                    cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
                    p_count += 1
        else:
            for b in p_res[0].boxes.xyxy:
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                p_count += 1

    # 2. Vehicle detections
    v_res = vehicle_model(left, conf=0.25, verbose=False)
    v_count = 0
    for box in v_res[0].boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            v_count += 1
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(vis, vehicle_model.names[cls_id].upper(), (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # 3. Banner
    banner_h = 48
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)
    cv2.putText(banner, title, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, f"{meta_str} | Vehicles: {v_count} | Potholes: {p_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

    out = np.vstack([banner, vis])
    return out


def build_candidate_contact_sheet():
    candidates = [
        {
            "name": "Candidate 1: Main Street Urban Stereo",
            "meta": "Duration: 23.1s | 640x360 @ 60 FPS SBS | ZED/Stereo",
            "path": PROJECT_ROOT / "temp_candidates" / "main_street_sbs.mp4",
            "is_video": True,
            "frame_idx": 300
        },
        {
            "name": "Candidate 2: Highway Driving Stereo",
            "meta": "Duration: 15.9s | 960x540 @ 60 FPS SBS | Dual-Cam",
            "path": PROJECT_ROOT / "temp_candidates" / "driving_full_sbs_16s.mp4",
            "is_video": True,
            "frame_idx": 200
        },
        {
            "name": "Candidate 3: Speed Hump / Pothole Road",
            "meta": "Duration: 54.3s | 640x336 @ 30 FPS SBS | Dashcam",
            "path": PROJECT_ROOT / "temp_candidates" / "gitc_speed_hump.mp4",
            "is_video": True,
            "frame_idx": 300
        },
        {
            "name": "Candidate 4: PavementDamagesG7 ZED 2i",
            "meta": "Continuous Sequence | 1920x1080 @ 30 FPS | ZED 2i Left+Right",
            "path": PROJECT_ROOT / "data" / "PavementDamagesG7" / "extracted" / "IMAGES" / "img_16-10-05-465220_i.jpg",
            "is_video": False,
            "frame_idx": 0
        },
    ]

    target_w = 640
    target_h = 360
    rendered_panels = []

    for c in candidates:
        if not c["path"].exists():
            print(f"Skipping missing: {c['path']}")
            continue
        left, right = get_candidate_frame(c["path"], c["is_video"], c["frame_idx"])
        if left is None:
            continue
        left_resized = cv2.resize(left, (target_w, target_h))
        annotated = annotate_frame(left_resized, c["name"], c["meta"])
        rendered_panels.append(annotated)

    if len(rendered_panels) < 4:
        print("Not enough candidate panels")
        return

    # 2x2 grid
    row1 = np.hstack([rendered_panels[0], rendered_panels[1]])
    row2 = np.hstack([rendered_panels[2], rendered_panels[3]])
    grid = np.vstack([row1, row2])

    out_path = PROJECT_ROOT / "outputs" / "demo" / "vehicle_pothole_candidate_contact_sheet.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"Saved contact sheet to {out_path}")

    # Copy to artifact dir
    artifact_copy = ARTIFACT_DIR / "vehicle_pothole_candidate_contact_sheet.png"
    shutil.copy(str(out_path), str(artifact_copy))
    print(f"Copied to artifact dir: {artifact_copy}")


if __name__ == "__main__":
    build_candidate_contact_sheet()
