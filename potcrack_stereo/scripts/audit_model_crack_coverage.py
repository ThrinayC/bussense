#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit_model_crack_coverage.py
=====================================
STEP 1 & STEP 2: Comprehensive Crack Coverage & Confidence Threshold Audit
- Evaluates combined_pothole_crack model across target frames: [41, 43, 44, 47, 48, 49, 65, 66]
- Control clean frames: [0, 10, 20]
- Sweep confidence levels: [0.005, 0.010, 0.015, 0.020, 0.025, 0.035]
- Measures: crack boxes, mask pixels, coverage %, connected components
- Generates side-by-side visual overlays to inspect useful coverage vs background noise.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=== BUSSENSE CRACK COVERAGE & CONFIDENCE THRESHOLD AUDIT ===")
    
    model_path = PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    
    data_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "left"
    out_dir = PROJECT_ROOT / "outputs" / "debug" / "crack_coverage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    target_frames = [41, 43, 44, 47, 48, 49, 65, 66]
    clean_frames = [0, 10, 20]
    conf_levels = [0.005, 0.010, 0.015, 0.020, 0.025, 0.035]

    results = {"target_frames": {}, "clean_frames": {}}

    # 1. Clean Frame Controls
    print("\n--- CONTROL CLEAN FRAMES (False-Positive Check) ---")
    for fnum in clean_frames:
        fpath = data_dir / f"frame_{fnum:05d}.jpg"
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        H, W = img.shape[:2]
        total_px = H * W
        frame_res = {}

        for conf in conf_levels:
            res = model.predict(img, conf=conf, iou=0.90, verbose=False)[0]
            union_mask = np.zeros((H, W), dtype=np.uint8)
            box_count = 0

            if res.boxes is not None and res.masks is not None:
                for i, box in enumerate(res.boxes):
                    if int(box.cls[0].item()) != 1:
                        continue
                    b_coords = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    if b_coords[1] < 70:
                        continue
                    box_area = (b_coords[2] - b_coords[0]) * (b_coords[3] - b_coords[1])
                    if box_area > 0.35 * total_px:
                        continue
                    m_raw = res.masks.data[i].cpu().numpy()
                    m_res = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    if cv2.countNonZero(m_res) > 0.05 * total_px:
                        continue
                    union_mask = cv2.bitwise_or(union_mask, m_res * 255)
                    box_count += 1

            mask_px = int(cv2.countNonZero(union_mask))
            num_cc, _ = cv2.connectedComponents((union_mask > 0).astype(np.uint8))
            frame_res[str(conf)] = {
                "boxes": box_count,
                "pixels": mask_px,
                "coverage_pct": round(mask_px / total_px * 100.0, 4),
                "components": max(0, num_cc - 1)
            }
        results["clean_frames"][fnum] = frame_res
        print(f"Clean Frame {fnum:02d}: " + " | ".join([f"conf {c}: {frame_res[str(c)]['boxes']} boxes, {frame_res[str(c)]['pixels']} px" for c in [0.005, 0.010, 0.020, 0.035]]))

    # 2. Target Crack Frames
    print("\n--- TARGET CRACK FRAMES AUDIT ---")
    for fnum in target_frames:
        fpath = data_dir / f"frame_{fnum:05d}.jpg"
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        H, W = img.shape[:2]
        total_px = H * W
        frame_res = {}
        panel_images = []

        # Panel 0: Original image cropped or scaled
        vis_orig = img.copy()
        cv2.putText(vis_orig, f"FRAME {fnum:02d} ORIGINAL", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        panel_images.append(vis_orig)

        for conf in conf_levels:
            res = model.predict(img, conf=conf, iou=0.90, verbose=False)[0]
            union_mask = np.zeros((H, W), dtype=np.uint8)
            box_count = 0

            if res.boxes is not None and res.masks is not None:
                for i, box in enumerate(res.boxes):
                    if int(box.cls[0].item()) != 1:
                        continue
                    b_coords = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    if b_coords[1] < 70:
                        continue
                    box_area = (b_coords[2] - b_coords[0]) * (b_coords[3] - b_coords[1])
                    if box_area > 0.35 * total_px:
                        continue
                    m_raw = res.masks.data[i].cpu().numpy()
                    m_res = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    if cv2.countNonZero(m_res) > 0.05 * total_px:
                        continue
                    union_mask = cv2.bitwise_or(union_mask, m_res * 255)
                    box_count += 1

            mask_px = int(cv2.countNonZero(union_mask))
            num_cc, _ = cv2.connectedComponents((union_mask > 0).astype(np.uint8))
            frame_res[str(conf)] = {
                "boxes": box_count,
                "pixels": mask_px,
                "coverage_pct": round(mask_px / total_px * 100.0, 4),
                "components": max(0, num_cc - 1)
            }

            # Visual overlay panel
            vis_p = img.copy()
            overlay = vis_p.copy()
            overlay[union_mask > 0] = [0, 220, 255] # Yellow
            vis_p = cv2.addWeighted(vis_p, 0.4, overlay, 0.6, 0)
            cnts, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_p, cnts, -1, (0, 0, 255), 2)
            cv2.putText(vis_p, f"conf = {conf:.3f} | Boxes: {box_count:02d} | Pixels: {mask_px:,} ({mask_px/total_px*100.0:.2f}%) | CC: {max(0, num_cc - 1)}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            panel_images.append(vis_p)

        results["target_frames"][fnum] = frame_res
        print(f"Frame {fnum:02d}: " + " | ".join([f"conf {c}: {frame_res[str(c)]['boxes']}b, {frame_res[str(c)]['pixels']}px ({frame_res[str(c)]['coverage_pct']}%)" for c in conf_levels]))

        # Compose 7-panel contact sheet (Original + 6 conf levels: 2 rows of 3-4 panels or stacked)
        # Resize panels to 640x360
        thumbs = [cv2.resize(p, (640, 360)) for p in panel_images] # 7 images
        # Pad to 8 images with a blank telemetry panel
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"FRAME {fnum:02d} COVERAGE SUMMARY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_off = 100
        for c in conf_levels:
            r = frame_res[str(c)]
            cv2.putText(blank, f"conf {c:.3f}: {r['boxes']:02d} boxes | {r['pixels']:,} px ({r['coverage_pct']:.2f}%) | {r['components']} CC",
                        (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            y_off += 38
        thumbs.append(blank)

        row1 = np.hstack(thumbs[:4])
        row2 = np.hstack(thumbs[4:])
        contact_sheet = np.vstack([row1, row2])

        sheet_path = out_dir / f"crack_coverage_sweep_frame_{fnum:02d}.jpg"
        cv2.imwrite(str(sheet_path), contact_sheet)

    # Save JSON summary
    out_json = eval_dir / "model_crack_coverage_audit.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved audit summary JSON to: {out_json}")
    print(f"Saved contact sheets to: {out_dir}")

if __name__ == "__main__":
    main()
