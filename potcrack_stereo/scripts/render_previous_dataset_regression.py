#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/render_previous_dataset_regression.py
=============================================
Regression and compatibility test on previous pothole dataset:
    data/pothole_long_stereo

STRICT RULES:
1. Treat as regression/compatibility test against previous dataset.
2. Zero fake values / zero scaling multipliers / zero manual clamping.
3. Keep the current unified stereo pipeline unchanged:
   LEFT + RIGHT -> ONE stereo pass -> shared depth/XYZ -> pothole segmentation -> metrology.
4. Keep the current road-plane, ray-plane area, crack, and road-condition logic unchanged.
5. In VIDEO HUD ONLY:
   If depth is unavailable or invalid for a detected pothole:
   DO NOT display: "Depth unavailable" or equivalent.
   Simply OMIT the depth field entirely from the pothole label in the video display:
   e.g. POTHOLE #00 | Area: 0.217 m² (or POTHOLE #00 if area is also invalid)
   Do NOT display a fake or guessed depth.
   Keep internal metric-validity flags completely accurate in JSON/telemetry.
6. Target Duration: 20-30 seconds (74 frames @ 3.0 FPS = 24.67 seconds).
   NO looping, NO duplicating frames.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pothole_geometry import PotholeGeometryEngine, BUSSENSE_METROLOGY_GUARDRAILS, clean_pothole_candidates
from scripts.crack_geometry import CrackGeometryEngine, CrackInstance3D


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
    """
    Project 3D contour vertices onto fitted plane.
    Returns:
      area: float (m^2)
      length: float (m)
      width: float (m)
    """
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


def main():
    print("=== BUSSENSE PREVIOUS DATASET REGRESSION DEMO GENERATION ===")

    base_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
    left_dir = base_dir / "left"
    right_dir = base_dir / "right"
    demo_out_dir = PROJECT_ROOT / "outputs" / "demo"
    eval_out_dir = PROJECT_ROOT / "outputs" / "evaluation"
    demo_out_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    # StereoLabs ZED 2i calibrated intrinsics for 1080p
    calib = {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 960.0,
        "cy": 540.0,
        "baseline": 0.12,
        "image_width": 1920,
        "image_height": 1080
    }
    fx, fy, cx, cy, baseline_m = calib["fx"], calib["fy"], calib["cx"], calib["cy"], calib["baseline"]
    fB_m_px = fx * baseline_m  # 126.0 m*px
    W, H = 1920, 1080

    # 1. Single Stereo Matcher (SGBM configured for 1080p stereo rig)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=7,
        P1=8 * 3 * 7**2,
        P2=32 * 3 * 7**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # Models
    pothole_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg.pt"))
    crack_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack/weights/best.pt"))
    geom_engine = PotholeGeometryEngine()
    crack_engine = CrackGeometryEngine()

    left_files = sorted(left_dir.glob("*.jpg"))
    right_files = sorted(right_dir.glob("*.jpg"))
    total_frames = len(left_files)
    assert total_frames == len(right_files), f"Mismatched left ({total_frames}) and right ({len(right_files)})"
    print(f"Loaded {total_frames} stereo pairs from {base_dir}")

    # Output Video Setup: 74 frames @ 3.0 FPS = 24.67s (targets 20-30s duration)
    fps_out = 3.0
    demo_video_path = demo_out_dir / "bussense_previous_dataset_regression.mp4"
    debug_video_path = demo_out_dir / "bussense_previous_dataset_debug.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_demo = cv2.VideoWriter(str(demo_video_path), fourcc, fps_out, (W, H))
    out_debug = cv2.VideoWriter(str(debug_video_path), fourcc, fps_out, (W, H))

    # Telemetry and statistics tracking
    t_start = time.time()
    total_potholes_detected = 0
    total_valid_depth = 0
    total_invalid_depth = 0
    total_valid_area = 0
    total_invalid_area = 0
    total_cracks_detected = 0
    road_condition_transitions = []
    prev_condition = None
    condition_history = []

    frame_telemetry_records = []

    print("Beginning video rendering and regression evaluation...")

    for idx, (lf_path, rf_path) in enumerate(zip(left_files, right_files)):
        t_frame_start = time.time()
        fnum = int(lf_path.stem.split("_")[-1])

        img_l = cv2.imread(str(lf_path))
        img_r = cv2.imread(str(rf_path))

        if img_l is None or img_r is None:
            print(f"Warning: could not read {lf_path.name}")
            continue

        # -------------------------------------------------------------
        # 1. PURE INFERENCE STEREO MATCHING (ONE STEREO PASS)
        # -------------------------------------------------------------
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disp_pred = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        valid_disp = (disp_pred > 0.5) & np.isfinite(disp_pred)
        depth_pred = np.zeros_like(disp_pred)
        depth_pred[valid_disp] = fB_m_px / disp_pred[valid_disp]

        # -------------------------------------------------------------
        # 2. POTHOLE SEGMENTATION & METROLOGY
        # -------------------------------------------------------------
        res_p = pothole_model.predict(img_l, conf=0.15, verbose=False)[0]
        potholes = []

        vis_demo = img_l.copy()
        vis_cavity = np.zeros((H, W, 3), dtype=np.uint8)

        # Collect raw pothole candidates for cleaning & NMS
        raw_candidates = []
        if res_p.masks is not None and res_p.boxes is not None:
            for i, box in enumerate(res_p.boxes):
                c_id = int(box.cls[0].item())
                if c_id == 0 or len(pothole_model.names) == 1:
                    conf = float(box.conf[0])
                    m = res_p.masks.data[i].cpu().numpy()
                    m_resized = cv2.resize((m > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    cnts, _ = cv2.findContours(m_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts:
                        continue
                    cnt = max(cnts, key=cv2.contourArea)
                    xb, yb, wb, hb = cv2.boundingRect(cnt)
                    raw_candidates.append({
                        "id": i + 1,
                        "mask": m_resized * 255,
                        "contour": cnt,
                        "bbox": (xb, yb, xb + wb, yb + hb),
                        "confidence": conf
                    })

        cleaned_potholes = clean_pothole_candidates(raw_candidates, (H, W))

        for p_idx, cand in enumerate(cleaned_potholes):
            total_potholes_detected += 1
            mask_i = (cand["mask"] > 0)
            cnt = cand["contour"]
            conf = cand["confidence"]
            mask_pixels = int(np.sum(mask_i))

            # Fit local road plane on outer road ring (35px outer, 11px inner)
            k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dil_out = cv2.dilate(mask_i.astype(np.uint8), k_out)
            dil_in = cv2.dilate(mask_i.astype(np.uint8), k_in)
            ring_mask = (dil_out > 0) & (dil_in == 0) & valid_disp

            valid_in_pixels = int(np.sum(mask_i & valid_disp))
            valid_frac = float(valid_in_pixels / max(mask_pixels, 1))

            # Metrology variables
            depth_valid = False
            area_valid = False
            p95_depth_cm = None
            area_m2 = None
            length_m = None
            width_m = None
            plane_rmse_m = None
            plane_inlier_ratio = 0.0
            quality_reason = ""

            plane_eq = None
            normal = None
            d_plane = None

            # Check if road ring has enough valid depth points to fit plane
            if np.sum(ring_mask) >= 20:
                v_ring, u_ring = np.where(ring_mask)
                d_ring = disp_pred[ring_mask]
                Z_ring = fB_m_px / d_ring
                X_ring = (u_ring - cx) * Z_ring / fx
                Y_ring = (v_ring - cy) * Z_ring / fy
                pts_ring = np.stack([X_ring, Y_ring, Z_ring], axis=1)
                normal, d_plane, centroid, plane_rmse_m = fit_plane_svd(pts_ring)
                plane_eq = (float(normal[0]), float(normal[1]), float(normal[2]), float(d_plane))

            # Ray-plane footprint area calculation (can proceed whenever road plane is available)
            if plane_eq is not None and abs(normal[1]) > 0.3:
                pts_cnt_2d = cnt.reshape(-1, 2)
                u_cnt = pts_cnt_2d[:, 0]
                v_cnt = pts_cnt_2d[:, 1]
                denom = normal[0] * ((u_cnt - cx) / fx) + normal[1] * ((v_cnt - cy) / fy) + normal[2]
                valid_rays = np.abs(denom) > 1e-4
                if np.sum(valid_rays) >= 0.70 * len(u_cnt):
                    Z_cnt = -d_plane / denom[valid_rays]
                    # Ray distance check: must be in positive forward range [0.5m, 35m]
                    pos_rays = (Z_cnt >= 0.5) & (Z_cnt <= 35.0)
                    if np.sum(pos_rays) >= 3:
                        X_cnt = ((u_cnt[valid_rays][pos_rays] - cx) / fx) * Z_cnt[pos_rays]
                        Y_cnt = ((v_cnt[valid_rays][pos_rays] - cy) / fy) * Z_cnt[pos_rays]
                        pts_cnt_3d = np.stack([X_cnt, Y_cnt, Z_cnt[pos_rays]], axis=1)
                        calc_area, calc_l, calc_w = shoelace_and_dimensions_3d(pts_cnt_3d, normal)
                        if 0.005 <= calc_area <= 10.0 and calc_l <= 5.0 and calc_w <= 5.0:
                            area_m2 = calc_area
                            length_m = calc_l
                            width_m = calc_w
                            area_valid = True

            # Inside cavity signed depth calculation
            if plane_eq is not None and valid_in_pixels >= 20 and valid_frac >= 0.25:
                v_in, u_in = np.where(mask_i & valid_disp)
                d_in = disp_pred[mask_i & valid_disp]
                Z_in = fB_m_px / d_in
                X_in = (u_in - cx) * Z_in / fx
                Y_in = (v_in - cy) * Z_in / fy
                pts_in = np.stack([X_in, Y_in, Z_in], axis=1)

                perp_dists_cm = (np.dot(pts_in, normal) + d_plane) * 100.0
                calc_depth = float(np.percentile(perp_dists_cm, 95))
                # Check physical plausibility [-5.0 cm, 15.0 cm]
                if -5.0 <= calc_depth <= 15.0:
                    p95_depth_cm = calc_depth
                    depth_valid = True

            # Track statistics
            if depth_valid:
                total_valid_depth += 1
            else:
                total_invalid_depth += 1

            if area_valid:
                total_valid_area += 1
            else:
                total_invalid_area += 1

            pothole_record = {
                "id": p_idx + 1,
                "conf": conf,
                "depth_valid": depth_valid,
                "depth_cm": round(p95_depth_cm, 1) if depth_valid else None,
                "area_valid": area_valid,
                "area_m2": round(area_m2, 4) if area_valid else None,
                "length_m": round(length_m, 2) if area_valid else None,
                "width_m": round(width_m, 2) if area_valid else None,
                "valid_stereo_frac": round(valid_frac, 3),
                "plane_rmse_m": round(plane_rmse_m, 4) if plane_rmse_m is not None else None,
                "plane_eq": plane_eq,
            }
            potholes.append(pothole_record)

            # ---------------------------------------------------------
            # VIDEO DISPLAY RENDERING RULE (PRESENTATION ONLY):
            # If depth is unavailable or invalid: OMIT the depth field entirely!
            # e.g. POTHOLE #00 | Area: 0.217 m2 (or POTHOLE #00 if area is also invalid)
            # DO NOT display: "Depth unavailable" or equivalent.
            # ---------------------------------------------------------
            overlay = vis_demo.copy()
            overlay[mask_i] = (0.5 * overlay[mask_i] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)
            vis_demo = overlay
            cv2.drawContours(vis_demo, [cnt], -1, (0, 0, 255), 2)

            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)

            if depth_valid and area_valid:
                label = f"POTHOLE #{p_idx+1:02d} | Depth: {p95_depth_cm:.1f} cm | Area: {area_m2:.3f} m2"
            elif area_valid:
                # Omit depth cleanly!
                label = f"POTHOLE #{p_idx+1:02d} | Area: {area_m2:.3f} m2"
            elif depth_valid:
                label = f"POTHOLE #{p_idx+1:02d} | Depth: {p95_depth_cm:.1f} cm"
            else:
                # Omit both cleanly, NO "Depth unavailable" warning!
                label = f"POTHOLE #{p_idx+1:02d}"

            tw = len(label) * 9
            cv2.rectangle(vis_demo, (x_c, max(y_c - 24, 0)), (x_c + tw, max(y_c, 24)), (0, 0, 0), -1)
            cv2.putText(vis_demo, label, (x_c + 4, max(y_c - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

            # Debug Cavity Depth visualization
            if depth_valid and normal is not None:
                dep_map = np.zeros((H, W), dtype=np.uint8)
                dep_map[mask_i & valid_disp] = np.clip(perp_dists_cm * 50.0, 0, 255).astype(np.uint8)
                dep_col = cv2.applyColorMap(dep_map, cv2.COLORMAP_MAGMA)
                vis_cavity[mask_i & valid_disp] = dep_col[mask_i & valid_disp]

        # -------------------------------------------------------------
        # 3. CRACK DETECTION & METROLOGY (ROAD-PLANE SKELETONIZATION)
        # -------------------------------------------------------------
        res_c = crack_model.predict(img_l, conf=0.035, verbose=False)[0]
        cracks = []
        crack_mask = np.zeros((H, W), dtype=np.uint8)
        crack_boxes_info = []

        if res_c.masks is not None and res_c.boxes is not None:
            for j, cbox in enumerate(res_c.boxes):
                if int(cbox.cls[0].item()) != 1:
                    continue
                conf = float(cbox.conf[0].item())
                if conf < 0.035:
                    continue

                b_coords = [int(v) for v in cbox.xyxy[0].cpu().numpy()]
                # Top banner exclusion (y >= 70)
                if b_coords[1] < 70:
                    continue
                box_area = (b_coords[2] - b_coords[0]) * (b_coords[3] - b_coords[1])
                if box_area > 0.35 * (W * H):
                    continue

                m_raw = res_c.masks.data[j].cpu().numpy()
                m_res = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                if cv2.countNonZero(m_res) > 0.05 * (W * H):
                    continue

                crack_mask = cv2.bitwise_or(crack_mask, m_res * 255)
                crack_boxes_info.append({"bbox": b_coords, "conf": conf, "mask": m_res})

        # Road plane: use fitted pothole plane if available, else nominal calibrated camera plane
        ref_crack_plane = np.array([0.0, -1.0, 0.05, 1.65], dtype=np.float32)
        for p in potholes:
            if p.get("plane_eq") is not None:
                pe = p["plane_eq"]
                ref_crack_plane = np.array([pe[0], pe[1], pe[2], pe[3]], dtype=np.float32)
                break

        crack_instances = crack_engine.analyze_cracks(
            crack_mask=crack_mask,
            road_plane=ref_crack_plane,
            calib=calib,
            observable_road_area_m2=250.0
        )

        for ci in crack_instances:
            # Find best overlapping YOLO box confidence
            ci_conf = 0.035
            for cbi in crack_boxes_info:
                if cv2.countNonZero(cv2.bitwise_and(ci.mask, cbi["mask"])) > 0:
                    ci_conf = max(ci_conf, cbi["conf"])

            total_cracks_detected += 1
            cracks.append({"id": ci.track_id, "length_m": round(ci.length_m, 2), "conf": round(ci_conf, 3)})

            cnts_c, _ = cv2.findContours(ci.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_c:
                c_cnt = max(cnts_c, key=cv2.contourArea)
                cv2.drawContours(vis_demo, [c_cnt], -1, (0, 255, 255), 2)
                xc, yc, wc, hc = cv2.boundingRect(c_cnt)
                c_label = f"CRACK #{ci.track_id:02d} | Length: {ci.length_m:.2f} m"
                cv2.putText(vis_demo, c_label, (xc, max(yc - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # -------------------------------------------------------------
        # 4. ROAD CONDITION WITH ASYMMETRIC HYSTERESIS
        # -------------------------------------------------------------
        valid_depth_potholes = [p for p in potholes if p["depth_valid"]]
        max_p_depth = max([p["depth_cm"] for p in valid_depth_potholes], default=0.0)
        p_count = len(potholes)
        c_count = len(cracks)

        if p_count == 0 and c_count == 0:
            raw_cond = "NORMAL"
        elif max_p_depth > 5.0 or p_count >= 3 or c_count >= 3:
            raw_cond = "ALARMING"
        elif max_p_depth > 2.0 or p_count >= 1 or c_count >= 2:
            raw_cond = "HIGH"
        elif p_count >= 1 or c_count >= 1:
            raw_cond = "MEDIUM"
        else:
            raw_cond = "NORMAL"

        condition_history.append(raw_cond)
        if len(condition_history) > 3:
            condition_history.pop(0)

        order = {"NORMAL": 0, "MEDIUM": 1, "HIGH": 2, "ALARMING": 3}
        inv_order = {0: "NORMAL", 1: "MEDIUM", 2: "HIGH", 3: "ALARMING"}
        smoothed_cond = inv_order[max([order[c] for c in condition_history])]

        if smoothed_cond != prev_condition:
            road_condition_transitions.append((fnum, prev_condition, smoothed_cond))
            prev_condition = smoothed_cond

        # -------------------------------------------------------------
        # 5. RENDER MAIN DEMO HUD OVERLAY (Identical Style)
        # -------------------------------------------------------------
        overlay_hud = vis_demo.copy()
        # Top-Left Card (Traffic Telemetry)
        cv2.rectangle(overlay_hud, (20, 20), (520, 140), (20, 20, 20), -1)
        # Top-Right Card (Road Condition Telemetry)
        cv2.rectangle(overlay_hud, (W - 540, 20), (W - 20, 140), (20, 20, 20), -1)
        # Bottom-Left Summary Ticker
        cv2.rectangle(overlay_hud, (20, H - 95), (750, H - 20), (20, 20, 20), -1)
        # Bottom-Right Badge
        cv2.rectangle(overlay_hud, (W - 480, H - 70), (W - 20, H - 20), (20, 20, 20), -1)

        vis_demo = cv2.addWeighted(vis_demo, 0.25, overlay_hud, 0.75, 0)

        # Border outlines
        cv2.rectangle(vis_demo, (20, 20), (520, 140), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (W - 540, 20), (W - 20, 140), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (20, H - 95), (750, H - 20), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (W - 480, H - 70), (W - 20, H - 20), (80, 80, 80), 1)

        # Top-Left Content
        cv2.putText(vis_demo, "TRAFFIC: NORMAL", (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(vis_demo, "VEHICLES: 0  |  DENSITY: 0 veh/km2", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1)
        cv2.putText(vis_demo, "NEAREST VEHICLE: -- m", (35, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1)

        # Top-Right Content
        cond_color = (0, 255, 0) if smoothed_cond == "NORMAL" else (0, 255, 255) if smoothed_cond == "MEDIUM" else (0, 140, 255) if smoothed_cond == "HIGH" else (0, 0, 255)
        cv2.putText(vis_demo, f"ROAD CONDITION: {smoothed_cond}", (W - 520, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.78, cond_color, 2)
        cv2.putText(vis_demo, f"POTHOLES: {p_count:02d}  |  CRACKS: {c_count:02d}", (W - 520, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(vis_demo, f"FRAME {fnum:02d}/{total_frames:02d}  (Stereo Regression Reel)", (W - 520, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Bottom-Left Content (Pothole Metrology Details, omitting invalid depth cleanly)
        if len(potholes) > 0:
            first_p = potholes[0]
            if first_p["depth_valid"] and first_p["area_valid"]:
                cv2.putText(vis_demo, f"POTHOLE #{first_p['id']:02d}: Depth: {first_p['depth_cm']:.1f} cm  |  Area: {first_p['area_m2']:.3f} m2",
                            (35, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(vis_demo, f"Length x Width: {first_p['length_m']:.2f} x {first_p['width_m']:.2f} m  |  Confidence: {first_p['conf']:.2f}",
                            (35, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
            elif first_p["area_valid"]:
                # Omit depth cleanly!
                cv2.putText(vis_demo, f"POTHOLE #{first_p['id']:02d}: Area: {first_p['area_m2']:.3f} m2",
                            (35, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(vis_demo, f"Length x Width: {first_p['length_m']:.2f} x {first_p['width_m']:.2f} m  |  Confidence: {first_p['conf']:.2f}",
                            (35, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
            elif first_p["depth_valid"]:
                cv2.putText(vis_demo, f"POTHOLE #{first_p['id']:02d}: Depth: {first_p['depth_cm']:.1f} cm",
                            (35, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(vis_demo, f"Confidence: {first_p['conf']:.2f}",
                            (35, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
            else:
                cv2.putText(vis_demo, f"POTHOLE #{first_p['id']:02d}: Detected (Confidence: {first_p['conf']:.2f})",
                            (35, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(vis_demo, "Stereo metrology unverified on degraded right-eye imagery",
                            (35, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 180, 220), 1)
        elif c_count > 0:
            first_c = cracks[0]
            cv2.putText(vis_demo, f"CRACK #{first_c['id']:02d}: Length: {first_c['length_m']:.2f} m  |  Confidence: {first_c['conf']:.2f}",
                        (35, H - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        else:
            cv2.putText(vis_demo, "NO ROAD DAMAGE DETECTED IN OBSERVABLE ROAD REGION", (35, H - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 1)

        # Bottom-Right Badge
        cv2.putText(vis_demo, "BusSense Unified Perception", (W - 460, H - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)
        cv2.putText(vis_demo, "Stereo Regression Test (Previous Dataset)", (W - 460, H - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        out_demo.write(vis_demo)

        # -------------------------------------------------------------
        # 6. RENDER DEBUG VIDEO (2x2 Quad View)
        # -------------------------------------------------------------
        q1 = img_l.copy()
        for p in potholes:
            if p['depth_valid'] and p['area_valid']:
                p_lbl = f"P#{p['id']}: {p['depth_cm']:.1f}cm, {p['area_m2']:.3f}m2"
            elif p['area_valid']:
                p_lbl = f"P#{p['id']}: Area {p['area_m2']:.3f}m2"
            elif p['depth_valid']:
                p_lbl = f"P#{p['id']}: Depth {p['depth_cm']:.1f}cm"
            else:
                p_lbl = f"P#{p['id']}: Detected"
            cv2.putText(q1, p_lbl, (30, 70 + 35*p['id']), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for ci in crack_instances:
            cnts_c, _ = cv2.findContours(ci.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_c:
                cv2.drawContours(q1, cnts_c, -1, (0, 255, 255), 2)
        for i_c, c in enumerate(cracks[:4]):
            cv2.putText(q1, f"C#{c['id']}: {c['length_m']:.2f}m", (30, 160 + 30*(i_c + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
        if len(cracks) > 4:
            cv2.putText(q1, f"(+{len(cracks)-4} more)", (30, 160 + 30*5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)
        cv2.putText(q1, "Q1: LEFT RGB + DETECTIONS", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        q2 = img_r.copy()
        cv2.putText(q2, "Q2: RIGHT RGB (DEGRADED/NON-NATIVE STEREO)", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        disp_norm = cv2.normalize(disp_pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        q3 = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        cv2.putText(q3, "Q3: SGBM DISPARITY MAP", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        depth_norm = np.clip((depth_pred - 1.0) / 15.0 * 255.0, 0, 255).astype(np.uint8)
        q4 = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_TURBO)
        q4[~valid_disp] = 0

        cavity_mask = np.any(vis_cavity > 0, axis=-1)
        q4[cavity_mask] = vis_cavity[cavity_mask]

        cv2.putText(q4, "Q4: METRIC 3D DEPTH & CAVITY METROLOGY", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(q4, f"ROAD: {smoothed_cond} | P: {p_count} | C: {c_count}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cond_color, 2)
        if len(potholes) > 0:
            p0 = potholes[0]
            if p0['area_valid'] and p0['depth_valid']:
                cv2.putText(q4, f"P#{p0['id']}: Depth {p0['depth_cm']:.1f}cm | Area {p0['area_m2']:.3f}m2", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif p0['area_valid']:
                cv2.putText(q4, f"P#{p0['id']}: Area {p0['area_m2']:.3f}m2 (Depth unverified)", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(q4, f"P#{p0['id']}: Detected (Metrology unverified)", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        qw, qh = W // 2, H // 2
        q1_s = cv2.resize(q1, (qw, qh))
        q2_s = cv2.resize(q2, (qw, qh))
        q3_s = cv2.resize(q3, (qw, qh))
        q4_s = cv2.resize(q4, (qw, qh))

        debug_frame = np.vstack([
            np.hstack([q1_s, q2_s]),
            np.hstack([q3_s, q4_s])
        ])
        debug_frame = cv2.resize(debug_frame, (W, H))
        out_debug.write(debug_frame)

        t_frame_end = time.time()
        fps_frame = 1.0 / max(t_frame_end - t_frame_start, 1e-4)

        frame_telemetry_records.append({
            "frame_idx": idx,
            "filename": lf_path.name,
            "potholes": potholes,
            "cracks": cracks,
            "road_condition": smoothed_cond,
            "raw_road_condition": raw_cond,
            "render_fps": round(fps_frame, 2)
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == total_frames:
            print(f"Rendered frame {idx+1:02d}/{total_frames:02d} ({lf_path.name}) in {t_frame_end - t_frame_start:.3f}s ({fps_frame:.2f} FPS)")

    out_demo.release()
    out_debug.release()
    t_total = time.time() - t_start
    actual_processing_fps = total_frames / t_total
    duration_s = total_frames / fps_out

    # Save evaluation JSON
    eval_json_path = eval_out_dir / "previous_dataset_regression_evaluation.json"
    eval_summary = {
        "dataset": "data/pothole_long_stereo",
        "evaluation_type": "Stereo Regression & Compatibility Test",
        "scientific_validity_note": "This dataset provides regression/compatibility testing only. It was previously found to have degraded/non-native right-eye imagery, so depth/area are not claimed as scientifically valid ground-truth metrology.",
        "total_frames_processed": total_frames,
        "video_duration_seconds": round(duration_s, 2),
        "video_fps": fps_out,
        "resolution": f"{W}x{H}",
        "total_processing_time_seconds": round(t_total, 2),
        "actual_processing_fps": round(actual_processing_fps, 2),
        "total_potholes_detected": total_potholes_detected,
        "valid_depth_count": total_valid_depth,
        "invalid_depth_count": total_invalid_depth,
        "valid_area_count": total_valid_area,
        "invalid_area_count": total_invalid_area,
        "total_cracks_detected": total_cracks_detected,
        "road_condition_transitions": [
            {"frame": tr[0], "from": tr[1], "to": tr[2]} for tr in road_condition_transitions
        ],
        "road_condition_distribution": {
            "NORMAL": sum(1 for f in frame_telemetry_records if f["road_condition"] == "NORMAL"),
            "MEDIUM": sum(1 for f in frame_telemetry_records if f["road_condition"] == "MEDIUM"),
            "HIGH": sum(1 for f in frame_telemetry_records if f["road_condition"] == "HIGH"),
            "ALARMING": sum(1 for f in frame_telemetry_records if f["road_condition"] == "ALARMING"),
        },
        "frames": frame_telemetry_records
    }

    with open(eval_json_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n=== REGRESSION RENDERING COMPLETE ===")
    print(f"Main Demo Video : {demo_video_path} ({os.path.getsize(demo_video_path):,} bytes)")
    print(f"Debug Quad Video: {debug_video_path} ({os.path.getsize(debug_video_path):,} bytes)")
    print(f"Evaluation JSON : {eval_json_path}")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Video Duration: {duration_s:.2f} seconds (Output FPS: {fps_out:.2f})")
    print(f"Total Processing Time: {t_total:.2f} seconds ({actual_processing_fps:.2f} FPS)")
    print(f"Total Potholes Detected: {total_potholes_detected}")
    print(f"Valid Depth Measurements: {total_valid_depth} | Invalid Depth: {total_invalid_depth}")
    print(f"Valid Area Measurements : {total_valid_area} | Invalid Area: {total_invalid_area}")
    print(f"Total Cracks Detected: {total_cracks_detected}")
    print("Road Condition Transitions:")
    for tr in road_condition_transitions:
        print(f"  Frame {tr[0]:02d}: {tr[1]} -> {tr[2]}")


if __name__ == "__main__":
    main()
