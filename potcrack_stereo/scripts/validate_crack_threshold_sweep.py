#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_crack_threshold_sweep.py
=========================================
Runs full 74-frame validation sweep across conf = 0.035, 0.05, 0.08.
Tests:
- Total crack detections
- Frames with cracks vs zero cracks
- Max and median confidence
- Total and average crack lengths (m)
- Behavior on clean, nominal, transition, and cracked sections
- False positive inspection
- Generates before/after comparison frames
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.crack_geometry import CrackGeometryEngine

def main():
    print("=== FULL 74-FRAME CRACK THRESHOLD VALIDATION SWEEP ===")
    
    out_dir = PROJECT_ROOT / "outputs" / "debug" / "crack_web_regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    left_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "left"
    frames = sorted(left_dir.glob("*.jpg"))
    total_frames = len(frames)
    assert total_frames == 74, f"Expected 74 frames, found {total_frames}"
    
    model_path = PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    
    calib = {"fx": 1050.0, "fy": 1050.0, "cx": 960.0, "cy": 540.0, "baseline": 0.12}
    ref_road_plane = np.array([0.0, -1.0, 0.05, 1.65], dtype=np.float32)
    crack_engine = CrackGeometryEngine()
    
    thresholds = [0.035, 0.05, 0.08]
    sweep_results = {str(th): {
        "total_detections": 0,
        "frames_with_cracks": 0,
        "frames_without_cracks": 0,
        "all_confs": [],
        "all_lengths_m": [],
        "frame_records": []
    } for th in thresholds}
    
    # Specific before/after debug frames to save
    debug_frames = {
        "clean": 0,
        "nominal": 20,
        "transition_in": 36,
        "alligator_1": 49,
        "transition_out": 52,
        "alligator_2": 65
    }
    
    for idx, f_path in enumerate(frames):
        img = cv2.imread(str(f_path))
        H, W = img.shape[:2]
        f_num = int(f_path.stem.split("_")[-1])
        
        # Raw inference down to 0.01 once per frame, then filter by threshold
        res = model.predict(img, conf=0.01, verbose=False)[0]
        
        for th in thresholds:
            crack_mask = np.zeros((H, W), dtype=np.uint8)
            inst_confs = []
            
            if res.boxes is not None and res.masks is not None:
                for b_idx, box in enumerate(res.boxes):
                    if int(box.cls[0].item()) != 1:
                        continue
                    conf = float(box.conf[0].item())
                    if conf < th:
                        continue
                    
                    b_coords = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    # Top banner exclusion (y >= 70)
                    if b_coords[1] < 70:
                        continue
                    box_area = (b_coords[2] - b_coords[0]) * (b_coords[3] - b_coords[1])
                    if box_area > 0.35 * (W * H):
                        continue
                        
                    m_raw = res.masks.data[b_idx].cpu().numpy()
                    m_res = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    if cv2.countNonZero(m_res) > 0.05 * (W * H):
                        continue
                        
                    crack_mask = cv2.bitwise_or(crack_mask, m_res * 255)
                    inst_confs.append(conf)
            
            instances = crack_engine.analyze_cracks(
                crack_mask=crack_mask,
                road_plane=ref_road_plane,
                calib=calib,
                observable_road_area_m2=250.0
            )
            
            c_cnt = len(instances)
            c_len = sum(ci.length_m for ci in instances)
            
            sw = sweep_results[str(th)]
            sw["total_detections"] += c_cnt
            if c_cnt > 0:
                sw["frames_with_cracks"] += 1
            else:
                sw["frames_without_cracks"] += 1
            sw["all_confs"].extend(inst_confs)
            sw["all_lengths_m"].extend([ci.length_m for ci in instances])
            
            sw["frame_records"].append({
                "frame": f_num,
                "count": c_cnt,
                "total_len_m": round(c_len, 2),
                "max_conf": round(max(inst_confs), 3) if inst_confs else 0.0
            })
            
            # Save visual comparison for specific debug frames
            if f_num in debug_frames.values():
                d_name = [k for k, v in debug_frames.items() if v == f_num][0]
                vis_img = img.copy()
                for ci in instances:
                    cnts, _ = cv2.findContours(ci.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cv2.drawContours(vis_img, [cnts[0]], -1, (0, 255, 255), 2)
                        bx = ci.bbox_xyxy
                        cv2.rectangle(vis_img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 200, 255), 1)
                        cv2.putText(vis_img, f"C#{ci.track_id:02d} {ci.length_m:.1f}m", (bx[0], max(bx[1]-4, 18)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                
                header_text = f"FRAME {f_num:02d} ({d_name.upper()}) | conf={th} | CRACKS: {c_cnt:02d} | LEN: {c_len:.2f}m"
                cv2.rectangle(vis_img, (20, 20), (W - 20, 65), (20, 20, 20), -1)
                cv2.putText(vis_img, header_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.imwrite(str(out_dir / f"compare_{d_name}_conf_{th}_frame_{f_num:02d}.jpg"), vis_img)

        if (idx + 1) % 10 == 0 or (idx + 1) == total_frames:
            print(f"Processed frame {idx+1:02d}/{total_frames:02d}")

    # Aggregated Summary
    summary_report = {}
    for th in thresholds:
        sw = sweep_results[str(th)]
        confs = sw["all_confs"]
        lens = sw["all_lengths_m"]
        summary_report[str(th)] = {
            "threshold": th,
            "total_crack_instances": sw["total_detections"],
            "frames_with_cracks": sw["frames_with_cracks"],
            "frames_without_cracks": sw["frames_without_cracks"],
            "max_confidence": round(float(np.max(confs)), 3) if len(confs) > 0 else 0.0,
            "median_confidence": round(float(np.median(confs)), 3) if len(confs) > 0 else 0.0,
            "mean_confidence": round(float(np.mean(confs)), 3) if len(confs) > 0 else 0.0,
            "total_length_meters": round(float(np.sum(lens)), 2) if len(lens) > 0 else 0.0,
            "mean_length_meters": round(float(np.mean(lens)), 2) if len(lens) > 0 else 0.0,
            "median_length_meters": round(float(np.median(lens)), 2) if len(lens) > 0 else 0.0,
            "clean_nominal_frames_with_detections": sum(1 for r in sw["frame_records"] if r["frame"] < 35 and r["count"] > 0),
            "cracked_frames_with_detections": sum(1 for r in sw["frame_records"] if (36 <= r["frame"] <= 50 or r["frame"] >= 61) and r["count"] > 0)
        }

    # Save to JSON
    summary_path = out_dir / "crack_threshold_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "summary": summary_report,
            "detailed_frames": {str(th): sweep_results[str(th)]["frame_records"] for th in thresholds}
        }, f, indent=2)

    print("\n" + "=" * 90)
    print("THRESHOLD SWEEP COMPARISON SUMMARY (FULL 74 FRAMES)")
    print("=" * 90)
    print(f"{'Metric':<38} | {'conf=0.035':^15} | {'conf=0.050':^15} | {'conf=0.080':^15}")
    print("-" * 90)
    s35 = summary_report["0.035"]
    s50 = summary_report["0.05"]
    s80 = summary_report["0.08"]
    print(f"{'Total Crack Instances Detected':<38} | {s35['total_crack_instances']:^15} | {s50['total_crack_instances']:^15} | {s80['total_crack_instances']:^15}")
    print(f"{'Frames with Cracks (out of 74)':<38} | {s35['frames_with_cracks']:^15} | {s50['frames_with_cracks']:^15} | {s80['frames_with_cracks']:^15}")
    print(f"{'Frames without Cracks (out of 74)':<38} | {s35['frames_without_cracks']:^15} | {s50['frames_without_cracks']:^15} | {s80['frames_without_cracks']:^15}")
    print(f"{'Max Confidence':<38} | {s35['max_confidence']:^15.3f} | {s50['max_confidence']:^15.3f} | {s80['max_confidence']:^15.3f}")
    print(f"{'Median Confidence':<38} | {s35['median_confidence']:^15.3f} | {s50['median_confidence']:^15.3f} | {s80['median_confidence']:^15.3f}")
    print(f"{'Total Metric Length (meters)':<38} | {s35['total_length_meters']:^15.1f} | {s50['total_length_meters']:^15.1f} | {s80['total_length_meters']:^15.1f}")
    print(f"{'Clean/Nominal Frames with Detections':<38} | {s35['clean_nominal_frames_with_detections']:^15} | {s50['clean_nominal_frames_with_detections']:^15} | {s80['clean_nominal_frames_with_detections']:^15}")
    print(f"{'Cracked Section Frames with Detections':<38} | {s35['cracked_frames_with_detections']:^15} | {s50['cracked_frames_with_detections']:^15} | {s80['cracked_frames_with_detections']:^15}")
    print("=" * 90)

if __name__ == "__main__":
    main()
