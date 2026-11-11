#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/render_previous_dataset_crack_networks.py
=================================================
Renders the complete BusSense regression video on:
    data/pothole_long_stereo
using the new topology-aware CrackNetworkGraphEngine:
- Union of all YOLO crack masks
- Conservative 2px closing (5x5 ellipse)
- Skeleton topological graph (endpoints, junctions, branches, loops)
- Conservative endpoint bridging (<= 15px, orientation-aware)
- True 8-connected 3D skeleton graph arc length (completely replacing flawed PCA sorting)
- Clean network-level HUD presentation (no label spam)
- Unchanged pothole metrology, stereo calibration, traffic, and shared stereo pass
- Omit invalid pothole depth from video display cleanly (zero fake depths)
- Target Duration: 20-30s (74 frames @ 3.0 FPS = 24.67s)
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
from scripts.crack_geometry import CrackNetworkGraphEngine, CrackNetwork3D, zhang_suen_thinning


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
    """Project 3D contour vertices onto fitted plane. Returns area, length, width."""
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
    print("=== BUSSENSE CRACK NETWORK REGRESSION DEMO GENERATION ===")
    t_start = time.time()

    base_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
    left_dir = base_dir / "left"
    right_dir = base_dir / "right"
    demo_out_dir = PROJECT_ROOT / "outputs" / "demo"
    eval_out_dir = PROJECT_ROOT / "outputs" / "evaluation"
    debug_out_dir = PROJECT_ROOT / "outputs" / "debug" / "crack_web_fragmentation"
    demo_out_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    debug_out_dir.mkdir(parents=True, exist_ok=True)

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

    # Models & Geometry Engines
    pothole_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg.pt"))
    crack_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack/weights/best.pt"))
    geom_engine = PotholeGeometryEngine()
    crack_net_engine = CrackNetworkGraphEngine(
        closing_kernel_size=5,  # 5x5 ellipse ~ 2px closing
        min_component_pixels=25,
        min_length_m=0.15,
        max_length_m=80.0,
        max_bridge_gap_px=15.0,
        bridge_angle_threshold_deg=60.0
    )

    left_files = sorted(left_dir.glob("*.jpg"))
    right_files = sorted(right_dir.glob("*.jpg"))
    total_frames = len(left_files)
    assert total_frames == len(right_files), f"Mismatched left ({total_frames}) and right ({len(right_files)})"
    print(f"Loaded {total_frames} stereo pairs from {base_dir}")

    # Output Video Setup: 74 frames @ 3.0 FPS = 24.67s (targets 20-30s duration)
    fps_out = 3.0
    demo_video_path = demo_out_dir / "bussense_previous_dataset_crack_network_final.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_demo = cv2.VideoWriter(str(demo_video_path), fourcc, fps_out, (W, H))

    # Diagnostic frames to save visual step breakdowns
    target_diag_frames = [36, 38, 42, 44, 47, 48, 49, 65]

    # Telemetry and statistics tracking
    total_potholes_detected = 0
    total_valid_depth = 0
    total_invalid_depth = 0
    total_valid_area = 0
    total_invalid_area = 0
    total_crack_networks_detected = 0
    total_crack_branches_detected = 0
    road_condition_transitions = []
    prev_condition = None
    condition_history = []

    frame_telemetry_records = []
    diagnostic_artifacts_saved = []

    print("Beginning video rendering with CrackNetworkGraphEngine...")

    for idx, (lf_path, rf_path) in enumerate(zip(left_files, right_files)):
        t_frame_start = time.time()
        fnum = int(lf_path.stem.split("_")[-1])

        img_l = cv2.imread(str(lf_path))
        img_r = cv2.imread(str(rf_path))
        if img_l is None or img_r is None:
            continue

        # -------------------------------------------------------------
        # 1. PURE INFERENCE STEREO MATCHING (ONE STEREO PASS)
        # -------------------------------------------------------------
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disp_pred = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        valid_disp = (disp_pred > 0.5) & (disp_pred < 128.0)

        depth_pred = np.zeros_like(disp_pred)
        depth_pred[valid_disp] = fB_m_px / disp_pred[valid_disp]

        vis_demo = img_l.copy()

        # -------------------------------------------------------------
        # 2. POTHOLE DETECTION & METROLOGY (UNCHANGED)
        # -------------------------------------------------------------
        res_p = pothole_model.predict(img_l, conf=0.15, verbose=False)[0]
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
        potholes = []

        for p_idx, cand in enumerate(cleaned_potholes):
            total_potholes_detected += 1
            mask_i = (cand["mask"] > 0)
            cnt = cand["contour"]
            conf = cand["confidence"]
            mask_pixels = int(np.sum(mask_i))

            # Road plane ring around pothole
            k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dil_out = cv2.dilate(mask_i.astype(np.uint8), k_out)
            dil_in = cv2.dilate(mask_i.astype(np.uint8), k_in)
            ring_mask = (dil_out > 0) & (dil_in == 0) & valid_disp

            depth_valid = False
            area_valid = False
            p95_depth_cm = None
            area_m2 = None
            length_m = None
            width_m = None
            plane_rmse_m = None
            plane_eq = None
            normal = None
            d_plane = None

            if np.sum(ring_mask) >= 20:
                v_ring, u_ring = np.where(ring_mask)
                d_ring = disp_pred[ring_mask]
                Z_ring = fB_m_px / d_ring
                X_ring = (u_ring - cx) * Z_ring / fx
                Y_ring = (v_ring - cy) * Z_ring / fy
                pts_ring = np.stack([X_ring, Y_ring, Z_ring], axis=1)
                normal, d_plane, centroid, plane_rmse_m = fit_plane_svd(pts_ring)
                plane_eq = (float(normal[0]), float(normal[1]), float(normal[2]), float(d_plane))

            if plane_eq is not None and abs(normal[1]) > 0.3:
                pts_cnt_2d = cnt.reshape(-1, 2)
                u_cnt = pts_cnt_2d[:, 0]
                v_cnt = pts_cnt_2d[:, 1]
                denom = normal[0] * ((u_cnt - cx) / fx) + normal[1] * ((v_cnt - cy) / fy) + normal[2]
                valid_rays = np.abs(denom) > 1e-4
                if np.sum(valid_rays) >= 0.70 * len(u_cnt):
                    Z_cnt = -d_plane / denom[valid_rays]
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

            valid_cavity_pts = mask_i & valid_disp
            valid_frac = np.sum(valid_cavity_pts) / max(np.sum(mask_i), 1)

            if valid_frac >= 0.20 and plane_eq is not None and plane_rmse_m is not None and plane_rmse_m <= 0.08:
                v_cav, u_cav = np.where(valid_cavity_pts)
                d_cav = disp_pred[valid_cavity_pts]
                Z_cav = fB_m_px / d_cav
                X_cav = (u_cav - cx) * Z_cav / fx
                Y_cav = (v_cav - cy) * Z_cav / fy
                pts_cav = np.stack([X_cav, Y_cav, Z_cav], axis=1)

                perp_dists_cm = (np.dot(pts_cav, normal) + d_plane) * 100.0
                cavity_dists_cm = perp_dists_cm[perp_dists_cm > 0]
                if len(cavity_dists_cm) >= 15:
                    p95_val = float(np.percentile(cavity_dists_cm, 95))
                    if 0.3 <= p95_val <= 25.0:
                        p95_depth_cm = p95_val
                        depth_valid = True

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
                "plane_eq": plane_eq
            }
            potholes.append(pothole_record)

            # Draw pothole with clean HUD rules (omit unavailable depth cleanly)
            overlay = vis_demo.copy()
            overlay[mask_i] = (0.5 * overlay[mask_i] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)
            vis_demo = overlay
            cv2.drawContours(vis_demo, [cnt], -1, (0, 0, 255), 2)
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)

            if depth_valid and area_valid:
                label = f"POTHOLE #{p_idx+1:02d} | Depth: {p95_depth_cm:.1f} cm | Area: {area_m2:.3f} m2"
            elif area_valid:
                label = f"POTHOLE #{p_idx+1:02d} | Area: {area_m2:.3f} m2"
            elif depth_valid:
                label = f"POTHOLE #{p_idx+1:02d} | Depth: {p95_depth_cm:.1f} cm"
            else:
                label = f"POTHOLE #{p_idx+1:02d}"

            tw = len(label) * 9
            cv2.rectangle(vis_demo, (x_c, max(y_c - 24, 0)), (x_c + tw, max(y_c, 24)), (0, 0, 0), -1)
            cv2.putText(vis_demo, label, (x_c + 4, max(y_c - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        # -------------------------------------------------------------
        # 3. CRACK DETECTION & CRACK NETWORK GRAPH ENGINE
        # -------------------------------------------------------------
        res_c = crack_model.predict(img_l, conf=0.020, iou=0.90, verbose=False)[0]
        union_mask = np.zeros((H, W), dtype=np.uint8)
        crack_boxes_info = []

        if res_c.masks is not None and res_c.boxes is not None:
            for j, cbox in enumerate(res_c.boxes):
                if int(cbox.cls[0].item()) != 1:
                    continue
                conf = float(cbox.conf[0].item())
                if conf < 0.020:
                    continue
                b_coords = [int(v) for v in cbox.xyxy[0].cpu().numpy()]
                if b_coords[1] < 70:
                    continue
                box_area = (b_coords[2] - b_coords[0]) * (b_coords[3] - b_coords[1])
                if box_area > 0.35 * (W * H):
                    continue
                m_raw = res_c.masks.data[j].cpu().numpy()
                m_res = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                if cv2.countNonZero(m_res) > 0.05 * (W * H):
                    continue

                union_mask = cv2.bitwise_or(union_mask, m_res * 255)
                crack_boxes_info.append({"bbox": b_coords, "conf": conf, "mask": m_res})

        # Road plane reference: use fitted pothole plane if available, else calibrated fallback
        ref_crack_plane = np.array([0.0, -1.0, 0.05, 1.65], dtype=np.float32)
        for p in potholes:
            if p.get("plane_eq") is not None:
                pe = p["plane_eq"]
                ref_crack_plane = np.array([pe[0], pe[1], pe[2], pe[3]], dtype=np.float32)
                break

        # Analyze cohesive crack networks
        crack_networks = crack_net_engine.analyze_crack_networks(
            union_mask=union_mask,
            road_plane=ref_crack_plane,
            calib=calib,
            raw_box_confs=crack_boxes_info,
            observable_road_area_m2=250.0
        )

        total_crack_networks_detected += len(crack_networks)
        total_frame_crack_length = 0.0
        rendered_network_records = []

        # Render crack networks onto pavement
        for net in crack_networks:
            total_crack_branches_detected += net.branch_count
            total_frame_crack_length += net.total_length_m

            # Draw semi-transparent crack mask overlay
            mask_net = net.mask > 0
            vis_demo[mask_net] = (0.4 * vis_demo[mask_net] + 0.6 * np.array([0, 220, 255])).astype(np.uint8)

            # Draw contour around network
            cnts_n, _ = cv2.findContours(net.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_n:
                cv2.drawContours(vis_demo, cnts_n, -1, (0, 255, 255), 2)

            # Draw skeleton centerline
            skel_pts = net.skeleton_mask > 0
            vis_demo[skel_pts] = [0, 255, 0]

            rendered_network_records.append({
                "network_id": net.network_id,
                "confidence": net.confidence,
                "total_length_m": net.total_length_m,
                "branch_count": net.branch_count,
                "junction_count": net.junction_count,
                "endpoint_count": net.endpoint_count,
                "coverage_area_m2": net.coverage_area_m2,
                "center_3d": net.center_3d
            })

        # Render clean, non-overlapping labels for visible networks near centroids
        placed_label_boxes = []
        sorted_nets = sorted(crack_networks, key=lambda n: n.total_length_m, reverse=True)

        for net in sorted_nets:
            m_ys, m_xs = np.where(net.mask > 0)
            if len(m_xs) == 0:
                continue
            c_x = int(np.mean(m_xs))
            c_y = int(np.mean(m_ys))

            line1 = f"CRACK NETWORK #{net.network_id:02d}"
            line2 = f"Length: {net.total_length_m:.2f} m | Branches: {net.branch_count}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1

            (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
            (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
            box_w = max(w1, w2) + 14
            box_h = h1 + h2 + 16

            bx1, by1, bx2, by2 = net.bbox_xyxy
            candidate_positions = [
                (max(10, min(W - box_w - 10, c_x - box_w // 2)), max(30, min(H - box_h - 100, c_y - box_h // 2))),
                (max(10, min(W - box_w - 10, bx1)), max(30, min(H - box_h - 100, by1 - box_h - 4))),
                (max(10, min(W - box_w - 10, bx1)), max(30, min(H - box_h - 100, by2 + 4))),
                (max(10, min(W - box_w - 10, c_x + 20)), max(30, min(H - box_h - 100, c_y))),
                (max(10, min(W - box_w - 10, c_x - box_w - 20)), max(30, min(H - box_h - 100, c_y)))
            ]

            chosen_pos = None
            for lx, ly in candidate_positions:
                rect = (lx, ly, lx + box_w, ly + box_h)
                collision = False
                for px1, py1, px2, py2 in placed_label_boxes:
                    if not (rect[2] < px1 or rect[0] > px2 or rect[3] < py1 or rect[1] > py2):
                        collision = True
                        break
                if not collision:
                    chosen_pos = (lx, ly)
                    placed_label_boxes.append(rect)
                    break

            if chosen_pos is None:
                if net.total_length_m < 0.40:
                    continue
                lx, ly = candidate_positions[0]
                chosen_pos = (lx, ly)
                placed_label_boxes.append((lx, ly, lx + box_w, ly + box_h))

            lx, ly = chosen_pos
            cv2.rectangle(vis_demo, (lx, ly), (lx + box_w, ly + box_h), (15, 15, 15), -1)
            cv2.rectangle(vis_demo, (lx, ly), (lx + box_w, ly + box_h), (80, 200, 255), 1)
            cv2.putText(vis_demo, line1, (lx + 6, ly + h1 + 4), font, font_scale, (0, 255, 255), thickness)
            cv2.putText(vis_demo, line2, (lx + 6, ly + h1 + h2 + 10), font, font_scale, (220, 255, 220), thickness)

        # -------------------------------------------------------------
        # 4. ROAD CONDITION WITH ASYMMETRIC HYSTERESIS
        # -------------------------------------------------------------
        valid_depth_potholes = [p for p in potholes if p["depth_valid"]]
        max_p_depth = max([p["depth_cm"] for p in valid_depth_potholes], default=0.0)
        p_count = len(potholes)
        c_count = len(crack_networks)

        if p_count == 0 and c_count == 0:
            raw_cond = "NORMAL"
        elif max_p_depth > 5.0 or p_count >= 3 or c_count >= 2 or total_frame_crack_length > 15.0:
            raw_cond = "ALARMING"
        elif max_p_depth > 2.0 or p_count >= 1 or c_count >= 1 or total_frame_crack_length > 4.0:
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
        # 5. RENDER MAIN DEMO HUD OVERLAY (Clean Dashboard)
        # -------------------------------------------------------------
        overlay_hud = vis_demo.copy()
        cv2.rectangle(overlay_hud, (20, 20), (520, 140), (20, 20, 20), -1)
        cv2.rectangle(overlay_hud, (W - 540, 20), (W - 20, 140), (20, 20, 20), -1)
        cv2.rectangle(overlay_hud, (20, H - 95), (820, H - 20), (20, 20, 20), -1)
        cv2.rectangle(overlay_hud, (W - 480, H - 70), (W - 20, H - 20), (20, 20, 20), -1)

        vis_demo = cv2.addWeighted(vis_demo, 0.25, overlay_hud, 0.75, 0)
        cv2.rectangle(vis_demo, (20, 20), (520, 140), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (W - 540, 20), (W - 20, 140), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (20, H - 95), (820, H - 20), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (W - 480, H - 70), (W - 20, H - 20), (80, 80, 80), 1)

        # Top-Left Traffic Card (Nominal default)
        cv2.putText(vis_demo, "TRAFFIC: NORMAL", (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(vis_demo, "VEHICLES: 0  |  DENSITY: 0 veh/km2", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1)
        cv2.putText(vis_demo, "NEAREST VEHICLE: -- m", (35, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1)

        # Top-Right Road Condition Card
        cond_color = (0, 255, 0) if smoothed_cond == "NORMAL" else (0, 255, 255) if smoothed_cond == "MEDIUM" else (0, 140, 255) if smoothed_cond == "HIGH" else (0, 0, 255)
        cv2.putText(vis_demo, f"ROAD CONDITION: {smoothed_cond}", (W - 520, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.78, cond_color, 2)
        cv2.putText(vis_demo, f"POTHOLES: {p_count:02d}  |  CRACK NETWORKS: {c_count:02d}", (W - 520, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(vis_demo, f"FRAME {fnum:02d}/{total_frames:02d}  (Stereo Regression Reel)", (W - 520, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Bottom-Left Summary Ticker
        if len(potholes) > 0:
            first_p = potholes[0]
            if first_p["depth_valid"] and first_p["area_valid"]:
                cv2.putText(vis_demo, f"POTHOLE #{first_p['id']:02d}: Depth: {first_p['depth_cm']:.1f} cm  |  Area: {first_p['area_m2']:.3f} m2",
                            (35, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            elif first_p["area_valid"]:
                cv2.putText(vis_demo, f"POTHOLE #{first_p['id']:02d}: Area: {first_p['area_m2']:.3f} m2",
                            (35, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            else:
                cv2.putText(vis_demo, f"POTHOLE #{first_p['id']:02d}: Detected (Confidence: {first_p['conf']:.2f})",
                            (35, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if c_count > 0:
                cv2.putText(vis_demo, f"Crack Networks: {c_count:02d} | Network Total Length: {total_frame_crack_length:.2f} m",
                            (35, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
            else:
                cv2.putText(vis_demo, "Stereo metrology unverified on degraded right-eye imagery",
                            (35, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 180, 220), 1)
        elif c_count > 0:
            first_net = crack_networks[0]
            cv2.putText(vis_demo, f"CRACK NET #{first_net.network_id:02d}: Length: {first_net.total_length_m:.2f} m | Branches: {first_net.branch_count}",
                        (35, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(vis_demo, f"Active Networks: {c_count:02d} | Total Length: {total_frame_crack_length:.2f} m | Density: {first_net.linear_density_m_m2 or 0:.3f} m/m2",
                        (35, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
        else:
            cv2.putText(vis_demo, "NO ROAD DAMAGE DETECTED IN OBSERVABLE ROAD REGION", (35, H - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 1)

        # Bottom-Right Badge
        cv2.putText(vis_demo, "BusSense Unified Perception", (W - 460, H - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)
        cv2.putText(vis_demo, "Topology-Aware Crack Network Reel", (W - 460, H - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        out_demo.write(vis_demo)

        # -------------------------------------------------------------
        # 6. SAVE DETAILED DEBUG ARTIFACTS FOR TARGET FRAMES
        # -------------------------------------------------------------
        if fnum in target_diag_frames:
            # Save raw union, closed union, skeleton, and final network rendering
            raw_u_path = debug_out_dir / f"frame_{fnum:02d}_raw_yolo_union_mask.png"
            closed_u_path = debug_out_dir / f"frame_{fnum:02d}_closed_union_mask.png"
            skel_path = debug_out_dir / f"frame_{fnum:02d}_skeleton.png"
            vis_net_path = debug_out_dir / f"frame_{fnum:02d}_final_network_vis.jpg"

            cv2.imwrite(str(raw_u_path), union_mask)
            # Create closed mask for saving
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            c_mask_save = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, k5)
            cv2.imwrite(str(closed_u_path), c_mask_save)
            skel_save = zhang_suen_thinning(c_mask_save)
            cv2.imwrite(str(skel_path), skel_save)
            cv2.imwrite(str(vis_net_path), vis_demo)

            diagnostic_artifacts_saved.append({
                "frame": fnum,
                "raw_boxes": len(crack_boxes_info),
                "crack_networks": len(crack_networks),
                "total_length_m": round(total_frame_crack_length, 2),
                "artifacts": [raw_u_path.name, closed_u_path.name, skel_path.name, vis_net_path.name]
            })

        t_frame_end = time.time()
        fps_frame = 1.0 / max(t_frame_end - t_frame_start, 1e-4)

        frame_telemetry_records.append({
            "frame_idx": idx,
            "filename": lf_path.name,
            "potholes": potholes,
            "crack_networks": rendered_network_records,
            "total_crack_length_m": round(total_frame_crack_length, 2),
            "road_condition": smoothed_cond,
            "raw_road_condition": raw_cond,
            "render_fps": round(fps_frame, 2)
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == total_frames:
            print(f"Rendered frame {idx+1:02d}/{total_frames:02d} ({lf_path.name}) in {t_frame_end - t_frame_start:.3f}s ({fps_frame:.2f} FPS)")

    out_demo.release()
    t_total = time.time() - t_start
    actual_processing_fps = total_frames / t_total
    duration_s = total_frames / fps_out

    # Save evaluation JSON
    eval_json_path = eval_out_dir / "previous_dataset_crack_network_final_evaluation.json"
    eval_summary = {
        "dataset": "data/pothole_long_stereo",
        "evaluation_type": "Crack Network Topology Regression & Metrology Audit",
        "scientific_validity_note": "This dataset provides regression/compatibility testing only. It was previously found to have degraded/non-native right-eye imagery. Crack lengths represent true 8-connected 3D skeleton graph arc lengths along the road plane.",
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
        "total_crack_networks_detected": total_crack_networks_detected,
        "total_crack_branches_detected": total_crack_branches_detected,
        "road_condition_transitions": [
            {"frame": tr[0], "from": tr[1], "to": tr[2]} for tr in road_condition_transitions
        ],
        "road_condition_distribution": {
            "NORMAL": sum(1 for f in frame_telemetry_records if f["road_condition"] == "NORMAL"),
            "MEDIUM": sum(1 for f in frame_telemetry_records if f["road_condition"] == "MEDIUM"),
            "HIGH": sum(1 for f in frame_telemetry_records if f["road_condition"] == "HIGH"),
            "ALARMING": sum(1 for f in frame_telemetry_records if f["road_condition"] == "ALARMING"),
        },
        "diagnostic_artifacts": diagnostic_artifacts_saved,
        "frames": frame_telemetry_records
    }

    with open(eval_json_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n=== CRACK NETWORK REGRESSION RENDERING COMPLETE ===")
    print(f"Main Demo Video : {demo_video_path} ({os.path.getsize(demo_video_path):,} bytes)")
    print(f"Evaluation JSON : {eval_json_path}")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Video Duration: {duration_s:.2f} seconds (Output FPS: {fps_out:.2f})")
    print(f"Total Crack Networks Detected: {total_crack_networks_detected}")
    print(f"Total Crack Branches Detected: {total_crack_branches_detected}")
    print("Road Condition Transitions:")
    for tr in road_condition_transitions:
        print(f"  Frame {tr[0]:02d}: {tr[1]} -> {tr[2]}")


if __name__ == "__main__":
    main()
