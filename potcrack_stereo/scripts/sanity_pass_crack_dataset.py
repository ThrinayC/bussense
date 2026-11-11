#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/sanity_pass_crack_dataset.py
====================================
Quick full-dataset sanity pass over data/pothole_long_stereo
using candidate production settings:
- crack confidence = 0.020
- segmentation IoU = 0.90
- CrackNetworkGraphEngine (simplified topological graph)
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.crack_geometry import CrackNetworkGraphEngine

def main():
    base_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "left"
    model_path = PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"
    model = YOLO(str(model_path))

    calib = {
        "fx": 1050.0, "fy": 1050.0, "cx": 960.0, "cy": 540.0,
        "baseline": 0.12, "image_width": 1920, "image_height": 1080
    }
    road_plane = np.array([0.0, -1.0, 0.05, 1.65], dtype=np.float32)

    engine = CrackNetworkGraphEngine(
        closing_kernel_size=5,
        min_component_pixels=25,
        min_length_m=0.15,
        max_length_m=80.0,
        max_bridge_gap_px=15.0,
        bridge_angle_threshold_deg=60.0
    )

    left_files = sorted(base_dir.glob("*.jpg"))
    total_frames = len(left_files)
    print(f"Running sanity pass on {total_frames} frames...")

    frames_with_cracks = 0
    total_networks = 0
    network_counts_per_frame = []
    longest_network = {"frame": -1, "network_id": -1, "length_m": 0.0, "branches": 0}
    clean_section_detections = [] # frames 0-25

    for idx, fpath in enumerate(left_files):
        fnum = int(fpath.stem.split("_")[-1])
        img = cv2.imread(str(fpath))
        H, W = img.shape[:2]

        res = model.predict(img, conf=0.020, iou=0.90, verbose=False)[0]
        union_mask = np.zeros((H, W), dtype=np.uint8)
        crack_boxes_info = []

        if res.masks is not None and res.boxes is not None:
            for j, cbox in enumerate(res.boxes):
                if int(cbox.cls[0].item()) != 1:
                    continue
                conf = float(cbox.conf[0].item())
                b_coords = [int(v) for v in cbox.xyxy[0].cpu().numpy()]
                if b_coords[1] < 70:
                    continue
                box_area = (b_coords[2] - b_coords[0]) * (b_coords[3] - b_coords[1])
                if box_area > 0.35 * (W * H):
                    continue
                m_raw = res.masks.data[j].cpu().numpy()
                m_res = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                if cv2.countNonZero(m_res) > 0.05 * (W * H):
                    continue

                union_mask = cv2.bitwise_or(union_mask, m_res * 255)
                crack_boxes_info.append({"bbox": b_coords, "conf": conf, "mask": m_res})

        networks = engine.analyze_crack_networks(
            union_mask=union_mask,
            road_plane=road_plane,
            calib=calib,
            raw_box_confs=crack_boxes_info,
            observable_road_area_m2=250.0
        )

        n_count = len(networks)
        network_counts_per_frame.append(n_count)
        if n_count > 0:
            frames_with_cracks += 1
            total_networks += n_count
            for net in networks:
                if net.total_length_m > longest_network["length_m"]:
                    longest_network = {
                        "frame": fnum,
                        "network_id": net.network_id,
                        "length_m": round(net.total_length_m, 2),
                        "branches": net.branch_count
                    }
        if fnum <= 25 and n_count > 0:
            clean_section_detections.append((fnum, n_count))

    median_count = float(np.median(network_counts_per_frame))
    max_count = int(np.max(network_counts_per_frame))

    summary = {
        "total_frames": total_frames,
        "frames_with_crack_networks": frames_with_cracks,
        "total_networks": total_networks,
        "max_networks_per_frame": max_count,
        "median_network_count_per_frame": median_count,
        "longest_network": longest_network,
        "clean_section_detections": clean_section_detections
    }

    out_json = PROJECT_ROOT / "outputs" / "evaluation" / "crack_sanity_pass_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SANITY PASS RESULTS ===")
    print(f"total frames: {total_frames}")
    print(f"frames with crack networks: {frames_with_cracks}")
    print(f"total networks: {total_networks}")
    print(f"max networks/frame: {max_count}")
    print(f"median network count/frame: {median_count}")
    print(f"longest network: {longest_network['length_m']} m (Branches: {longest_network['branches']}) in frame {longest_network['frame']}")
    print(f"clean section detections (frames 0-25): {clean_section_detections}")

if __name__ == "__main__":
    main()
