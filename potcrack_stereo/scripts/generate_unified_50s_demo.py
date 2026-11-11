#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — 50-Second Unified Perception Regression Demo Generator
=================================================================
Executes the approved UnifiedBusSensePipeline over the 50-second KITTI multi-drive
stereo sequence (500 frames across 5 drives @ 10 FPS).

Produces:
1. outputs/demo/bussense_unified_50s_regression.mp4 (Full-screen Left RGB + Unified HUD)
2. outputs/demo/bussense_unified_50s_debug.mp4 (2x2 Quad Diagnostic View)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_unified_pipeline import UnifiedBusSensePipeline, UnifiedFrameState
from scripts.make_kitti_traffic_demo import (
    discover_drives, parse_kitti_calib, colorize_disparity, colorize_depth,
    KITTI_ROOT, CALIB_FILE
)

DEMO_DIR = PROJECT_ROOT / "outputs" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DEMO = DEMO_DIR / "bussense_unified_50s_regression.mp4"
TARGET_DEBUG = DEMO_DIR / "bussense_unified_50s_debug.mp4"

TARGET_FPS = 10
TARGET_FRAMES = 500  # Exactly 50.0 seconds
OUT_W, OUT_H = 1242, 375


def draw_unified_hud(
    img: np.ndarray,
    state: UnifiedFrameState,
    total_frames: int,
    fps: float,
    drive_name: str
):
    """Draws full-width unified inspection HUD banner on the image."""
    w = img.shape[1]
    banner_h = 74
    sub = img[0:banner_h, 0:w]
    bg = np.full_like(sub, (14, 18, 24))
    img[0:banner_h, 0:w] = cv2.addWeighted(bg, 0.88, sub, 0.12, 0)
    cv2.line(img, (0, banner_h), (w, banner_h), (0, 200, 255), 2)

    # Line 1: Title & Unified Status
    cv2.putText(img, "BUSSENSE  ▶  UNIFIED STEREO PERCEPTION  ▶  KITTI EGO VEHICLE (MOVING)",
                (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    status_str = "STEREO: ACTIVE  |  VEHICLES: ACTIVE  |  POTHOLES: ACTIVE"
    cv2.putText(img, status_str, (w - 440, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 180), 1, cv2.LINE_AA)

    # Line 2: Traffic Metrics
    tm = state.traffic_metrics
    v_cnt = tm.get("visible_vehicle_count", len(state.vehicles))
    t_cnt = tm.get("tracked_vehicle_count", len(state.vehicles))
    near_dist = tm.get("nearest_vehicle_distance_m")
    near_cls = tm.get("nearest_vehicle_class", "")
    near_gap = tm.get("nearest_longitudinal_gap_m")
    dens_val = tm.get("traffic_density_veh_km2")
    dens_stat = tm.get("density_status", "UNRELIABLE")

    near_str = f"{near_dist:.1f}m ({near_cls})" if near_dist is not None else "NONE"
    gap_str = f"{near_gap:.1f}m" if near_gap is not None else "N/A"
    dens_str = f"{dens_val:.0f} veh/km²" if (dens_stat == "OK" and dens_val is not None) else "UNRELIABLE"
    dens_col = (0, 255, 180) if dens_stat == "OK" else (80, 100, 255)

    telem1 = f"TRAFFIC: {v_cnt} (Tracked: {t_cnt})  |  NEAREST: {near_str}  |  GAP [HEURISTIC]: {gap_str}  |  DENSITY: "
    cv2.putText(img, telem1, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200, 220, 240), 1, cv2.LINE_AA)
    (tw1, _), _ = cv2.getTextSize(telem1, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
    cv2.putText(img, dens_str, (12 + tw1, 45), cv2.FONT_HERSHEY_DUPLEX, 0.38, dens_col, 1, cv2.LINE_AA)

    # Line 3: Pothole & Road Metrics
    p_cnt = len(state.potholes)
    if p_cnt > 0:
        max_d = max([p.get("depth_p95_cm", 0.0) or 0.0 for p in state.potholes])
        tot_a = sum([p.get("area_m2", 0.0) or 0.0 for p in state.potholes])
        pot_str = f"POTHOLES: {p_cnt}  |  MAX DEPTH: {max_d:.1f}cm  |  DAMAGE AREA: {tot_a:.4f}m²"
        pot_col = (0, 165, 255)
    else:
        pot_str = "POTHOLES: 0 (ROAD SURFACE NOMINAL)"
        pot_col = (180, 200, 220)

    rp = state.road_plane
    rp_str = f"PLANE: {rp['source'].upper()} (H={rp['road_height_m']}m)" if (rp and rp.get("is_valid")) else "PLANE: SEARCHING"
    telem2 = f"{pot_str}  |  {rp_str}  |  DISP VALID: {state.stereo_quality['valid_disp_fraction']*100:.1f}%"
    cv2.putText(img, telem2, (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.36, pot_col, 1, cv2.LINE_AA)

    # Bottom footer strip
    h_img = img.shape[0]
    bot_sub = img[h_img - 20:h_img, 0:w]
    bot_bg = np.full_like(bot_sub, (12, 14, 18))
    img[h_img - 20:h_img, 0:w] = cv2.addWeighted(bot_bg, 0.85, bot_sub, 0.15, 0)
    time_sec = state.frame_id / max(1.0, fps)
    info = (f"Frame: {state.frame_id:04d}/{total_frames:04d}  |  T: {time_sec:.1f}s  |"
            f"  Drive: {drive_name}  |  Stereo Only  —  Zero LiDAR / Zero GT at Inference")
    cv2.putText(img, info, (12, h_img - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 175, 190), 1, cv2.LINE_AA)


def draw_vehicle_overlay(img: np.ndarray, veh: Dict[str, Any], sx: float, sy: float):
    h_img, w_img = img.shape[:2]
    b = veh["bbox"]
    bx1 = int(max(0, min(b[0] * sx, w_img - 1)))
    by1 = int(max(0, min(b[1] * sy, h_img - 1)))
    bx2 = int(max(0, min(b[2] * sx, w_img - 1)))
    by2 = int(max(0, min(b[3] * sy, h_img - 1)))

    box_color = (0, 230, 100)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), box_color, 2)

    u_bc, v_bc = (bx1 + bx2) // 2, by2
    cv2.circle(img, (u_bc, v_bc), 4, (0, 255, 255), -1)
    cv2.circle(img, (u_bc, v_bc), 6, (0, 180, 255), 1)

    disp_id = veh.get("display_id", f"Vehicle #{veh.get('track_id', 0)}")
    cls_name = veh.get("class", "vehicle").upper()
    dist_m = veh.get("distance_m")
    status = veh.get("status", "OK")
    X_m, Y_m, Z_m = veh.get("X_m"), veh.get("Y_m"), veh.get("Z_m")

    # Inspection card
    card_w, card_h = 230, 78
    card_x = max(10, min(bx1, w_img - card_w - 10))
    card_y = by1 - card_h - 8
    if card_y < 80:
        card_y = by2 + 8
    if card_y + card_h > h_img - 25:
        card_y = max(80, h_img - card_h - 25)

    sub = img[card_y:card_y + card_h, card_x:card_x + card_w]
    bg = np.full_like(sub, (20, 24, 30))
    img[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(bg, 0.85, sub, 0.15, 0)
    cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), box_color, 1)

    cv2.putText(img, f"{disp_id} [{cls_name}]", (card_x + 8, card_y + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)

    if status == "OK" and dist_m is not None:
        cv2.putText(img, f"Dist: {dist_m:.1f} m", (card_x + 8, card_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        if X_m is not None and Y_m is not None and Z_m is not None:
            cv2.putText(img, f"3D: ({X_m:+.1f}, {Y_m:+.1f}, {Z_m:.1f})m", (card_x + 8, card_y + 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 210, 240), 1, cv2.LINE_AA)
        cv2.putText(img, "Ref: Ray-Plane Ground Contact", (card_x + 8, card_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 170, 190), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "DISTANCE: UNRELIABLE", (card_x + 8, card_y + 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (80, 100, 255), 1, cv2.LINE_AA)


def draw_pothole_overlay(img: np.ndarray, pot: Dict[str, Any], sx: float, sy: float):
    b = pot["bbox"]
    x1 = int(b[0] * sx)
    y1 = int(b[1] * sy)
    x2 = int(b[2] * sx)
    y2 = int(b[3] * sy)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), 2)
    depth_cm = pot.get("depth_p95_cm", 0.0) or 0.0
    area_m2 = pot.get("area_m2", 0.0) or 0.0
    lbl = f"POT #{pot.get('track_id', 1)} | D={depth_cm:.1f}cm | A={area_m2:.3f}m²"
    cv2.putText(img, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)


def generate_unified_demo():
    print("\n" + "=" * 75)
    print(" BUSSENSE 50-SECOND UNIFIED REGRESSION DEMO GENERATION")
    print("=" * 75)

    # 1. Discover drives
    drives = discover_drives(KITTI_ROOT)
    calib = parse_kitti_calib(CALIB_FILE)

    all_left = []
    all_right = []
    all_names = []

    for drive_name, lefts, rights in drives:
        if len(all_left) >= TARGET_FRAMES:
            break
        rem = TARGET_FRAMES - len(all_left)
        take = min(len(lefts), rem)
        all_left += list(lefts[:take])
        all_right += list(rights[:take])
        all_names += [drive_name] * take
        print(f"  Adding {take} frames from {drive_name} (total: {len(all_left)})")

    n_frames = len(all_left)
    duration_s = n_frames / TARGET_FPS
    print(f"\n  Total Frames: {n_frames}  →  {duration_s:.1f}s @ {TARGET_FPS} fps")

    # 2. Initialize approved UnifiedBusSensePipeline
    pipeline = UnifiedBusSensePipeline(
        calib=calib,
        stereo_model="sgbm",
        conf_pothole=0.20,
        conf_vehicle=0.20
    )

    # 3. Video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    demo_writer = cv2.VideoWriter(str(TARGET_DEMO), fourcc, TARGET_FPS, (OUT_W, OUT_H))
    debug_writer = cv2.VideoWriter(str(TARGET_DEBUG), fourcc, TARGET_FPS, (OUT_W, OUT_H))

    if not demo_writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {TARGET_DEMO}")

    qw, qh = OUT_W // 2, OUT_H // 2
    prev_drive = None
    t_start = time.time()
    perf_records = []

    print(f"\n  Rendering 50-second unified demo...")
    print(f"  Output Demo  : {TARGET_DEMO}")
    print(f"  Output Debug : {TARGET_DEBUG}\n")

    for fi, (lp, rp, drive_name) in enumerate(zip(all_left, all_right, all_names)):
        if drive_name != prev_drive and prev_drive is not None:
            pipeline.reset()
            print(f"  ── Switched to drive: {drive_name} (tracker & states reset) ──")
        prev_drive = drive_name

        left_bgr = cv2.imread(str(lp))
        right_bgr = cv2.imread(str(rp))
        if left_bgr is None or right_bgr is None:
            continue
        if left_bgr.shape != right_bgr.shape:
            right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]))

        # Run ONE pass of UnifiedBusSensePipeline
        t_f0 = time.time()
        state = pipeline.process_frame(left_bgr, right_bgr, frame_index=fi + 1, timestamp=fi / float(TARGET_FPS))
        frame_time_ms = (time.time() - t_f0) * 1000.0
        perf_records.append(frame_time_ms)

        # -------------------------------------------------------------
        # Render Production Demo Frame
        # -------------------------------------------------------------
        demo_frame = cv2.resize(left_bgr, (OUT_W, OUT_H))
        sx = OUT_W / float(left_bgr.shape[1])
        sy = OUT_H / float(left_bgr.shape[0])

        # Draw vehicle overlays
        for veh in state.vehicles:
            draw_vehicle_overlay(demo_frame, veh, sx, sy)

        # Draw pothole overlays
        for pot in state.potholes:
            draw_pothole_overlay(demo_frame, pot, sx, sy)

        # Draw unified top HUD & bottom strip
        draw_unified_hud(demo_frame, state, n_frames, TARGET_FPS, drive_name[-15:])
        demo_writer.write(demo_frame)

        # -------------------------------------------------------------
        # Render 2x2 Debug Quad Frame
        # -------------------------------------------------------------
        # Q1: Left RGB with tracking
        q1 = cv2.resize(demo_frame, (qw, qh))
        cv2.putText(q1, "Q1: LEFT RGB + TRACKING + POTHOLES", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)

        # Q2: Right RGB with epipolar grid
        q2 = cv2.resize(right_bgr, (qw, qh))
        for line_y in range(30, qh, 40):
            cv2.line(q2, (0, line_y), (qw, line_y), (0, 100, 255), 1)
        cv2.putText(q2, "Q2: RIGHT RGB (EPIPOLAR CHECK)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1, cv2.LINE_AA)

        # Q3: Dense Disparity (Plasma)
        disp_color = colorize_disparity(state.disparity)
        q3 = cv2.resize(disp_color, (qw, qh))
        cv2.putText(q3, f"Q3: STEREO DISPARITY (Val: {state.stereo_quality['valid_disp_fraction']*100:.0f}%)",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Q4: Metric Depth (Viridis) + Telemetry
        depth_color = colorize_depth(state.depth_m)
        q4 = cv2.resize(depth_color, (qw, qh))
        cv2.putText(q4, f"Q4: METRIC DEPTH (Med: {state.stereo_quality['median_depth_m']:.1f}m)",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        debug_frame = np.vstack([np.hstack([q1, q2]), np.hstack([q3, q4])])
        debug_writer.write(debug_frame)

        if (fi + 1) % 25 == 0 or (fi + 1) == n_frames:
            el = time.time() - t_start
            fps_proc = (fi + 1) / max(0.001, el)
            eta = (n_frames - fi - 1) / max(0.001, fps_proc)
            v_cnt = len(state.vehicles)
            p_cnt = len(state.potholes)
            print(f"  [{fi+1:04d}/{n_frames}] veh={v_cnt} pot={p_cnt} | proc={fps_proc:.1f}fps | elapsed={el:.0f}s | ETA={eta:.0f}s")

    demo_writer.release()
    debug_writer.release()

    total_time = time.time() - t_start
    demo_sz_mb = TARGET_DEMO.stat().st_size / (1024 * 1024)
    debug_sz_mb = TARGET_DEBUG.stat().st_size / (1024 * 1024)
    avg_ms = float(np.mean(perf_records)) if perf_records else 0.0

    print("\n" + "=" * 75)
    print(" 50-SECOND UNIFIED DEMO GENERATION COMPLETE")
    print("=" * 75)
    print(f"  Video Path (Demo) : {TARGET_DEMO}")
    print(f"  Video Path (Debug): {TARGET_DEBUG}")
    print(f"  Duration          : {duration_s:.1f} seconds")
    print(f"  Target FPS        : {TARGET_FPS} FPS")
    print(f"  Frames Processed  : {n_frames} / {TARGET_FRAMES}")
    print(f"  Demo File Size    : {demo_sz_mb:.2f} MB")
    print(f"  Debug File Size   : {debug_sz_mb:.2f} MB")
    print(f"  Total Gen Time    : {total_time:.1f}s (Avg {avg_ms:.1f}ms/frame)")
    print("=" * 75 + "\n")

    return {
        "success": True,
        "demo_path": str(TARGET_DEMO),
        "debug_path": str(TARGET_DEBUG),
        "duration_s": round(duration_s, 1),
        "fps": TARGET_FPS,
        "frames_processed": n_frames,
        "file_size_mb": round(demo_sz_mb, 2)
    }


if __name__ == "__main__":
    res = generate_unified_demo()
