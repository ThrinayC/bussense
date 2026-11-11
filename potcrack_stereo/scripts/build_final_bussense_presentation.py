#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_final_bussense_presentation.py
============================================
Builds the FINAL BusSense Presentation Reel:
- PART A: ~50 seconds of KITTI moving-vehicle forward-facing stereo driving (active traffic sensing)
- TRANSITION: Explicit transition slate distinguishing KITTI driving from native stereo pothole metrology
- PART B: Authentic native-stereo pothole metrology from data/fan_road_surface/dataset_1

Outputs:
- outputs/demo/bussense_final_demo.mp4
- outputs/demo/bussense_final_demo_debug.mp4
- outputs/evaluation/bussense_final_demo_evaluation.json
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.crack_geometry import CrackNetworkGraphEngine

# Presentation Canvas Configuration
CANVAS_W = 1280
CANVAS_H = 540
TARGET_FPS = 10.0

def resize_letterbox(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scales and letterboxes image to fit target canvas without aspect ratio distortion."""
    h, w = img.shape[:2]
    scale = min(target_w / float(w), target_h / float(h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

def fit_plane_svd(pts: np.ndarray):
    """Fit 3D plane using SVD. Returns normal (a, b, c), d, centroid, and plane RMSE."""
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2, :]
    if normal[1] < 0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    dists = np.dot(pts, normal) + d
    rmse = float(np.sqrt(np.mean(dists**2)))
    return normal, d, centroid, rmse

def shoelace_and_dimensions_3d(pts_3d: np.ndarray, normal: np.ndarray):
    """Project 3D contour vertices onto fitted plane. Returns area (m2), length (m), width (m)."""
    if len(pts_3d) < 3:
        return 0.0, 0.0, 0.0
    u_axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, u_axis)) > 0.9:
        u_axis = np.array([0.0, 1.0, 0.0])
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    u_axis = np.cross(v_axis, normal)
    u_axis /= np.linalg.norm(u_axis)
    
    pts_2d = np.stack([np.dot(pts_3d, u_axis), np.dot(pts_3d, v_axis)], axis=1)
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    pts_2d_cv = pts_2d.astype(np.float32)
    rect = cv2.minAreaRect(pts_2d_cv)
    w_box, h_box = rect[1]
    length = float(max(w_box, h_box))
    width = float(min(w_box, h_box))
    return float(area), length, width

def colorize_disparity(disp: np.ndarray, max_disp: float = 160.0) -> np.ndarray:
    valid = (disp > 0.5) & np.isfinite(disp)
    d_val = np.clip(disp, 0.0, max_disp) / max_disp * 255.0
    color = cv2.applyColorMap(d_val.astype(np.uint8), cv2.COLORMAP_PLASMA)
    color[~valid] = (20, 20, 20)
    return color

def colorize_depth(depth_m: np.ndarray, max_d: float = 30.0) -> np.ndarray:
    valid = (depth_m >= 0.5) & (depth_m <= max_d) & np.isfinite(depth_m)
    depth_norm = np.zeros_like(depth_m, dtype=np.uint8)
    if np.any(valid):
        d_val = (1.0 - (np.clip(depth_m, 0.5, max_d) - 0.5) / (max_d - 0.5)) * 255.0
        depth_norm = d_val.astype(np.uint8)
    color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
    color[~valid] = (15, 15, 15)
    return color

def create_transition_slate(width: int, height: int, is_debug: bool = False) -> np.ndarray:
    """Creates a transition slate between KITTI driving and Native Pothole Metrology."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Dark gradient background
    for y in range(height):
        factor = 0.6 + 0.4 * (y / height)
        canvas[y, :] = [int(22 * factor), int(18 * factor), int(14 * factor)]
    
    # Frame borders and accent lines
    cv2.rectangle(canvas, (40, 30), (width - 40, height - 30), (60, 60, 60), 1)
    cv2.line(canvas, (40, 110), (width - 40, 110), (0, 200, 255), 2)
    cv2.line(canvas, (40, height - 90), (width - 40, height - 90), (0, 200, 255), 2)

    # Title texts
    cv2.putText(canvas, "BUSSENSE  ▶  DATASET & SENSING TRANSITION", (60, 75),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(canvas, "NATIVE STEREO POTHOLE METROLOGY", (60, 180),
                cv2.FONT_HERSHEY_DUPLEX, 1.25, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "AUTHENTIC SYNCHRONIZED VEHICULAR DUAL-CAMERA RIG", (60, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 220, 240), 1, cv2.LINE_AA)

    # Key tech highlights
    cv2.putText(canvas, "• STEREO SOURCE: Fan Road Surface Benchmark (dataset_1)", (60, 285),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "• SENSOR CALIBRATION: fx = fy = 698.80 px | Baseline = 119.59 mm", (60, 325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "• PHYSICAL METROLOGY: Local Road-Plane SVD  |  Signed P95 Cavity Depth  |  Ray-Plane Footprint Area", (60, 365),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 220, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "• STRICT AUDIT RULE: Pure Left+Right Stereo  |  NO GT / NO LiDAR / NO Monocular Depth at Inference", (60, 405),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255), 1, cv2.LINE_AA)

    status_tag = "DEBUG QUAD MODE: SYNCHRONIZED 2x2 PIPELINE INSPECTION" if is_debug else "MAIN REEL: UNIFIED FULL-SCREEN PERCEPTION HUD"
    cv2.putText(canvas, status_tag, (60, height - 52), cv2.FONT_HERSHEY_DUPLEX, 0.58, (180, 255, 180), 1, cv2.LINE_AA)

    return canvas


def main():
    print("============================================================")
    print(" BUSSENSE FINAL PRESENTATION VIDEO GENERATOR")
    print(" Part A: KITTI Stereo Driving (~50s)")
    print(" Transition: Native Stereo Slate (~2.5s)")
    print(" Part B: Authentic Stereo Pothole Metrology (~10.5s)")
    print("============================================================")

    t_start = time.time()
    demo_out_dir = PROJECT_ROOT / "outputs" / "demo"
    eval_out_dir = PROJECT_ROOT / "outputs" / "evaluation"
    demo_out_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    out_demo_path = demo_out_dir / "bussense_final_demo.mp4"
    out_debug_path = demo_out_dir / "bussense_final_demo_debug.mp4"
    out_eval_path = eval_out_dir / "bussense_final_demo_evaluation.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_demo = cv2.VideoWriter(str(out_demo_path), fourcc, TARGET_FPS, (CANVAS_W, CANVAS_H))
    writer_debug = cv2.VideoWriter(str(out_debug_path), fourcc, TARGET_FPS, (CANVAS_W, CANVAS_H))

    total_frames_rendered = 0
    kitti_frames_rendered = 0
    transition_frames_rendered = 0
    pothole_frames_rendered = 0

    # -------------------------------------------------------------
    # 1. PART A: KITTI STEREO DRIVING REEL (~50 SECONDS = 500 FRAMES)
    # -------------------------------------------------------------
    kitti_demo_src = demo_out_dir / "bussense_kitti_stereo_traffic_demo.mp4"
    kitti_debug_src = demo_out_dir / "bussense_kitti_stereo_traffic_debug.mp4"

    assert kitti_demo_src.exists(), f"Missing KITTI source: {kitti_demo_src}"
    assert kitti_debug_src.exists(), f"Missing KITTI debug source: {kitti_debug_src}"

    cap_kitti_demo = cv2.VideoCapture(str(kitti_demo_src))
    cap_kitti_debug = cv2.VideoCapture(str(kitti_debug_src))

    print("\n[PART A] Assembling KITTI Stereo Driving Sequence...")
    while True:
        ret_demo, frame_demo = cap_kitti_demo.read()
        ret_debug, frame_debug = cap_kitti_debug.read()
        if not ret_demo or not ret_debug:
            break

        # Standardize canvas to CANVAS_W x CANVAS_H (letterbox preserves native 1242x374 aspect ratio)
        canvas_demo = resize_letterbox(frame_demo, CANVAS_W, CANVAS_H)
        canvas_debug = resize_letterbox(frame_debug, CANVAS_W, CANVAS_H)

        writer_demo.write(canvas_demo)
        writer_debug.write(canvas_debug)
        kitti_frames_rendered += 1
        total_frames_rendered += 1

        if kitti_frames_rendered % 100 == 0:
            print(f"  KITTI Frames Rendered: {kitti_frames_rendered}/500 ({kitti_frames_rendered/TARGET_FPS:.1f}s)")

    cap_kitti_demo.release()
    cap_kitti_debug.release()
    print(f"  ✓ Part A Complete: {kitti_frames_rendered} frames ({kitti_frames_rendered/TARGET_FPS:.2f}s)")

    # -------------------------------------------------------------
    # 2. SOURCE TRANSITION SLATE (2.5 SECONDS = 25 FRAMES @ 10 FPS)
    # -------------------------------------------------------------
    print("\n[TRANSITION] Assembling Explicit Sensing Transition Slate...")
    trans_demo = create_transition_slate(CANVAS_W, CANVAS_H, is_debug=False)
    trans_debug = create_transition_slate(CANVAS_W, CANVAS_H, is_debug=True)

    n_trans_frames = 25  # 2.5 seconds
    for _ in range(n_trans_frames):
        writer_demo.write(trans_demo)
        writer_debug.write(trans_debug)
        transition_frames_rendered += 1
        total_frames_rendered += 1
    print(f"  ✓ Transition Slate Complete: {transition_frames_rendered} frames ({transition_frames_rendered/TARGET_FPS:.2f}s)")

    # -------------------------------------------------------------
    # 3. PART B: AUTHENTIC NATIVE-STEREO POTHOLE SEGMENT (~10.5s)
    # -------------------------------------------------------------
    print("\n[PART B] Processing Authentic Native-Stereo Potholes (Fan dataset_1)...")
    fan_dir = PROJECT_ROOT / "data" / "fan_road_surface" / "dataset_1"
    left_dir = fan_dir / "left"
    right_dir = fan_dir / "right"

    # Calibration parameters (Fan Native Stereo Rig)
    fx = 698.804085
    fy = 698.804085
    cx = 634.69008
    cy = 339.49033
    baseline_m = 0.119588
    fB_m_px = fx * baseline_m  # 83.56858 m*px

    stereo = cv2.StereoSGBM_create(
        minDisparity=48,
        numDisparities=160,
        blockSize=7,
        P1=8 * 7**2,
        P2=32 * 7**2,
        disp12MaxDiff=2,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    pothole_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg.pt"))
    crack_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"))
    crack_engine = CrackNetworkGraphEngine(closing_kernel_size=5, min_component_pixels=25)

    files = sorted([f for f in os.listdir(left_dir) if f.endswith(".png")], key=lambda x: int(os.path.splitext(x)[0]))
    total_pothole_pairs = len(files)
    print(f"  Loaded {total_pothole_pairs} continuous authentic stereo pairs.")

    pothole_eval_records = []
    total_valid_pothole_depths = 0
    total_valid_pothole_areas = 0

    qw, qh = CANVAS_W // 2, CANVAS_H // 2

    for p_idx, fname in enumerate(files):
        img_l = cv2.imread(str(left_dir / fname))
        img_r = cv2.imread(str(right_dir / fname))
        if img_l is None or img_r is None:
            continue
        H, W = img_l.shape[:2]

        # 1. Pure Inference Stereo Matching
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disp_pred = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        valid_disp = disp_pred > 50.0

        depth_pred = np.zeros_like(disp_pred)
        depth_pred[valid_disp] = fB_m_px / disp_pred[valid_disp]

        # 2. Pothole Detection & Metrology
        res_p = pothole_model.predict(img_l, conf=0.15, verbose=False)[0]
        vis_demo = img_l.copy()
        frame_potholes = []

        if res_p.boxes is not None and res_p.masks is not None:
            for i, box in enumerate(res_p.boxes):
                conf = float(box.conf[0])
                raw_mask = res_p.masks.data[i].cpu().numpy()
                mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask_pixels = int(np.sum(mask))
                if mask_pixels < 50:
                    continue

                # Ring plane fitting around pothole (outer 35px, inner 11px)
                k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
                k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                dil_out = cv2.dilate(mask.astype(np.uint8), k_out)
                dil_in = cv2.dilate(mask.astype(np.uint8), k_in)
                ring_mask = (dil_out > 0) & (dil_in == 0) & valid_disp

                valid_in_pixels = int(np.sum(mask & valid_disp))
                valid_frac = float(valid_in_pixels / max(mask_pixels, 1))

                cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) == 0:
                    continue
                main_cnt = max(cnts, key=cv2.contourArea)
                x_c, y_c, w_c, h_c = cv2.boundingRect(main_cnt)

                depth_valid = False
                area_valid = False
                p95_depth_cm = None
                area_m2 = None

                if np.sum(ring_mask) >= 40 and valid_in_pixels >= 20 and valid_frac >= 0.50:
                    v_ring, u_ring = np.where(ring_mask)
                    d_ring = disp_pred[ring_mask]
                    Z_ring = fB_m_px / d_ring
                    X_ring = (u_ring - cx) * Z_ring / fx
                    Y_ring = (v_ring - cy) * Z_ring / fy
                    pts_ring = np.stack([X_ring, Y_ring, Z_ring], axis=1)
                    normal, d_plane, centroid, plane_rmse_m = fit_plane_svd(pts_ring)

                    # Inside cavity signed depth
                    v_in, u_in = np.where(mask & valid_disp)
                    d_in = disp_pred[mask & valid_disp]
                    Z_in = fB_m_px / d_in
                    X_in = (u_in - cx) * Z_in / fx
                    Y_in = (v_in - cy) * Z_in / fy
                    pts_in = np.stack([X_in, Y_in, Z_in], axis=1)

                    perp_dists_cm = (np.dot(pts_in, normal) + d_plane) * 100.0
                    p95_depth_cm = float(np.percentile(perp_dists_cm, 95))
                    depth_valid = True
                    total_valid_pothole_depths += 1

                    # Footprint area via ray-plane intersection
                    pts_cnt_2d = main_cnt[:, 0, :]
                    u_cnt = pts_cnt_2d[:, 0]
                    v_cnt = pts_cnt_2d[:, 1]
                    denom = normal[0] * ((u_cnt - cx) / fx) + normal[1] * ((v_cnt - cy) / fy) + normal[2]
                    valid_rays = np.abs(denom) > 1e-4
                    if np.sum(valid_rays) >= 3:
                        Z_cnt = -d_plane / denom[valid_rays]
                        X_cnt = ((u_cnt[valid_rays] - cx) / fx) * Z_cnt
                        Y_cnt = ((v_cnt[valid_rays] - cy) / fy) * Z_cnt
                        pts_cnt_3d = np.stack([X_cnt, Y_cnt, Z_cnt], axis=1)
                        area_m2, length_m, width_m = shoelace_and_dimensions_3d(pts_cnt_3d, normal)
                        area_valid = True
                        total_valid_pothole_areas += 1

                # Visual overlay: Red fill + Contour
                overlay = vis_demo.copy()
                overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)
                vis_demo = overlay
                cv2.drawContours(vis_demo, [main_cnt], -1, (0, 0, 255), 2)

                # Strict formatting rule: Omit depth completely if invalid; never display "Depth unavailable"
                if depth_valid and area_valid:
                    label = f"POTHOLE #{i+1:02d} | Depth: {p95_depth_cm:.1f} cm | Area: {area_m2:.3f} m2"
                elif area_valid:
                    label = f"POTHOLE #{i+1:02d} | Area: {area_m2:.3f} m2"
                elif depth_valid:
                    label = f"POTHOLE #{i+1:02d} | Depth: {p95_depth_cm:.1f} cm"
                else:
                    label = f"POTHOLE #{i+1:02d}"

                tw = len(label) * 9
                cv2.rectangle(vis_demo, (x_c, max(y_c - 24, 0)), (x_c + tw, max(y_c, 24)), (0, 0, 0), -1)
                cv2.putText(vis_demo, label, (x_c + 4, max(y_c - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

                frame_potholes.append({
                    "id": i + 1, "conf": round(conf, 3), "depth_valid": depth_valid,
                    "depth_cm": round(p95_depth_cm, 2) if depth_valid else None,
                    "area_valid": area_valid, "area_m2": round(area_m2, 4) if area_valid else None
                })

        # 3. Crack Detection & Network Visualization
        res_c = crack_model.predict(img_l, conf=0.020, iou=0.90, verbose=False)[0]
        union_crack_mask = np.zeros((H, W), dtype=np.uint8)
        if res_c.boxes is not None and res_c.masks is not None:
            for j, cbox in enumerate(res_c.boxes):
                if int(cbox.cls[0].item()) != 1 or float(cbox.conf[0].item()) < 0.020:
                    continue
                m_raw = res_c.masks.data[j].cpu().numpy()
                m_res = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                union_crack_mask = cv2.bitwise_or(union_crack_mask, m_res * 255)

        ref_plane = np.array([0.0, -1.0, 0.05, 1.65], dtype=np.float32)
        crack_networks = crack_engine.analyze_crack_networks(
            union_mask=union_crack_mask,
            road_plane=ref_plane,
            calib={"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline_m}
        )

        for net in crack_networks:
            m_net = net.mask > 0
            vis_demo[m_net] = (0.4 * vis_demo[m_net] + 0.6 * np.array([0, 220, 255])).astype(np.uint8)
            sk_pts = net.skeleton_mask > 0
            vis_demo[sk_pts] = [0, 255, 0]

        # Top 3 crack labels
        for net in crack_networks[:3]:
            xc, yc = net.bbox_xyxy[0], net.bbox_xyxy[1]
            net_lbl = f"CRACK NET #{net.network_id:02d} | Length: {net.total_length_m:.2f} m | Branches: {net.branch_count}"
            tw_n = len(net_lbl) * 8
            cv2.rectangle(vis_demo, (xc, max(yc - 20, 0)), (xc + tw_n, max(yc, 20)), (15, 15, 15), -1)
            cv2.putText(vis_demo, net_lbl, (xc + 4, max(yc - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # 4. Standardized Presentation HUD Banner
        hud_overlay = vis_demo.copy()
        cv2.rectangle(hud_overlay, (20, 15), (580, 75), (20, 20, 20), -1)
        cv2.rectangle(hud_overlay, (W - 540, 15), (W - 20, 75), (20, 20, 20), -1)
        vis_demo = cv2.addWeighted(vis_demo, 0.35, hud_overlay, 0.65, 0)
        cv2.rectangle(vis_demo, (20, 15), (580, 75), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (W - 540, 15), (W - 20, 75), (80, 80, 80), 1)

        cv2.putText(vis_demo, "BUSSENSE  ▶  NATIVE STEREO POTHOLE METROLOGY", (32, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_demo, f"POTHOLES: {len(frame_potholes):02d}  |  CRACK NETWORKS: {len(crack_networks):02d}", (32, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

        p_valid_cnt = sum(1 for p in frame_potholes if p["depth_valid"])
        road_cond_str = "ALARMING" if p_valid_cnt > 0 else "HIGH" if len(frame_potholes) > 0 else "NORMAL"
        cond_col = (0, 0, 255) if road_cond_str == "ALARMING" else (0, 160, 255) if road_cond_str == "HIGH" else (0, 255, 0)
        cv2.putText(vis_demo, f"ROAD CONDITION: {road_cond_str}", (W - 520, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, cond_col, 2, cv2.LINE_AA)
        cv2.putText(vis_demo, f"FRAME {p_idx+1:02d}/{total_pothole_pairs:02d}  (Fan Native Stereo Rig)", (W - 520, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        # Main canvas
        canvas_demo = resize_letterbox(vis_demo, CANVAS_W, CANVAS_H)

        # 5. Debug Quad Canvas (Q1 Left, Q2 Right with Epipolar, Q3 Disparity, Q4 Depth)
        q1 = resize_letterbox(vis_demo, qw, qh)
        cv2.putText(q1, "Q1: LEFT RGB + POTHOLES + METROLOGY", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 255, 255), 1)

        r_annot = img_r.copy()
        for y_line in range(30, r_annot.shape[0], 50):
            cv2.line(r_annot, (0, y_line), (r_annot.shape[1], y_line), (0, 255, 0), 1)
        q2 = resize_letterbox(r_annot, qw, qh)
        cv2.putText(q2, "Q2: RIGHT RGB + EPIPOLAR REFERENCE LINES", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 255, 0), 1)

        disp_vis = colorize_disparity(disp_pred)
        q3 = resize_letterbox(disp_vis, qw, qh)
        cv2.putText(q3, "Q3: SHARED STEREO DISPARITY (PLASMA)", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 200, 0), 1)

        depth_vis = colorize_depth(depth_pred)
        q4 = resize_letterbox(depth_vis, qw, qh)
        cv2.putText(q4, "Q4: METRIC DEPTH Z=(f*B)/d (VIRIDIS)", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.40, (100, 255, 100), 1)

        quad_top = np.hstack([q1, q2])
        quad_bot = np.hstack([q3, q4])
        canvas_debug = np.vstack([quad_top, quad_bot])

        # Write frame repeated 3 times so 35 frames play at smooth presentation pace (~10.5 seconds @ 10 FPS)
        for _ in range(3):
            writer_demo.write(canvas_demo)
            writer_debug.write(canvas_debug)
            pothole_frames_rendered += 1
            total_frames_rendered += 1

        pothole_eval_records.append({
            "frame_idx": p_idx + 1,
            "filename": fname,
            "potholes": frame_potholes,
            "crack_networks_count": len(crack_networks)
        })

    writer_demo.release()
    writer_debug.release()

    total_time_s = time.time() - t_start
    duration_s = total_frames_rendered / TARGET_FPS

    # Build Evaluation Summary JSON
    eval_summary = {
        "presentation_title": "BusSense Autonomous Vehicle Road Perception Reel",
        "video_duration_seconds": round(duration_s, 2),
        "target_fps": TARGET_FPS,
        "total_frames_rendered": total_frames_rendered,
        "canvas_resolution": f"{CANVAS_W}x{CANVAS_H}",
        "sections": {
            "part_a_kitti_traffic": {
                "source": "KITTI Moving-Vehicle Forward-Facing Stereo (2011_09_26)",
                "drives_represented": [
                    "2011_09_26_drive_0002_sync",
                    "2011_09_26_drive_0005",
                    "2011_09_26_drive_0013_sync",
                    "2011_09_26_drive_0017_sync",
                    "2011_09_26_drive_0020_sync"
                ],
                "frames_count": kitti_frames_rendered,
                "duration_seconds": round(kitti_frames_rendered / TARGET_FPS, 2),
                "calibration": {
                    "fx": 721.54, "fy": 721.54, "cx": 609.56, "cy": 172.85, "baseline_m": 0.4706
                },
                "pipeline": "Left+Right -> SGBM -> Metric Depth -> YOLOv8 -> ByteTrack -> Ground-Contact 3D -> Traffic Density",
                "traffic_states_demonstrated": ["NORMAL", "MEDIUM", "HIGH"]
            },
            "transition": {
                "duration_seconds": round(transition_frames_rendered / TARGET_FPS, 2),
                "frames_count": transition_frames_rendered,
                "banner_text": "BUSSENSE — NATIVE STEREO POTHOLE METROLOGY | LEFT + RIGHT STEREO | NO GT / NO LiDAR / NO MONOCULAR DEPTH"
            },
            "part_b_native_stereo_potholes": {
                "source": "data/fan_road_surface/dataset_1",
                "stereo_type": "Authentic Synchronized Native Stereo Rig",
                "frames_count": pothole_frames_rendered,
                "unique_stereo_pairs": total_pothole_pairs,
                "duration_seconds": round(pothole_frames_rendered / TARGET_FPS, 2),
                "calibration": {
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline_m": baseline_m
                },
                "total_valid_pothole_depth_measurements": total_valid_pothole_depths,
                "total_valid_pothole_area_measurements": total_valid_pothole_areas,
                "metrology_pipeline": "SGBM Stereo -> Metric 3D -> Pothole Segmentation -> Local Road-Plane SVD -> Signed Depression Depth -> Ray-Plane Shoelace Area",
                "frames": pothole_eval_records
            }
        },
        "outputs": {
            "main_demo": str(out_demo_path),
            "debug_quad": str(out_debug_path),
            "file_size_main_mb": round(os.path.getsize(out_demo_path) / 1e6, 2),
            "file_size_debug_mb": round(os.path.getsize(out_debug_path) / 1e6, 2)
        }
    }

    with open(out_eval_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n============================================================")
    print(" FINAL BUSSENSE PRESENTATION REEL COMPLETE")
    print(f" Main Demo:   {out_demo_path} ({os.path.getsize(out_demo_path)/1e6:.2f} MB)")
    print(f" Debug Quad:  {out_debug_path} ({os.path.getsize(out_debug_path)/1e6:.2f} MB)")
    print(f" Evaluation:  {out_eval_path}")
    print(f" Total Time:  {total_time_s:.1f}s")
    print(f" Duration:    {duration_s:.2f}s ({total_frames_rendered} frames @ {TARGET_FPS} FPS)")
    print(f"   - KITTI Driving:     {kitti_frames_rendered/TARGET_FPS:.2f}s")
    print(f"   - Transition Slate:  {transition_frames_rendered/TARGET_FPS:.2f}s")
    print(f"   - Native Potholes:   {pothole_frames_rendered/TARGET_FPS:.2f}s")
    print("============================================================")

if __name__ == "__main__":
    main()
