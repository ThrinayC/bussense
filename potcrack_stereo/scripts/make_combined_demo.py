#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Phase 13 & 14: Combined Demo & Debug Video Generator
Generates:
1. outputs/demo/bussense_vehicle_pothole_demo.mp4 (15-30s full-screen inspection video)
2. outputs/demo/bussense_vehicle_pothole_debug.mp4 (2x2 synchronized quad diagnostic video)

Incorporates all 4 fixes:
- Fix 1: Explicit supported detector classes (car, truck, bus, motorcycle)
- Fix 2: Ground-contact position via camera ray / road plane intersection
- Fix 3: Explicit 3D road-plane polygon area calculation & UNRELIABLE guardrail
- Fix 4: SAME-LANE HEURISTIC explicit labeling
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_combined_pipeline import BusSensePerceptionPipeline

DEMO_DIR = PROJECT_ROOT / "outputs" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)


def draw_combined_hud_banner(
    img: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    traffic_scene: Dict[str, Any],
    density_telemetry: Dict[str, Any],
    potholes: List[Dict[str, Any]],
    s: float = 1.0
):
    """Draws sleek translucent top telemetry HUD banner."""
    w = img.shape[1]
    banner_h = int(68 * s)
    
    # Overlay banner
    sub = img[0:banner_h, 0:w]
    bg = np.full_like(sub, (16, 20, 26))
    img[0:banner_h, 0:w] = cv2.addWeighted(bg, 0.88, sub, 0.12, 0)
    cv2.line(img, (0, banner_h), (w, banner_h), (0, 200, 255), max(1, int(1.5 * s)))

    # Title Line
    title = "BUSSENSE STEREO INSPECTION & 3D ROAD METROLOGY"
    cv2.putText(img, title, (int(15 * s), int(24 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.44 * s, (255, 255, 255), max(1, int(1.2 * s)), cv2.LINE_AA)

    # Status pills on right of top line
    pothole_status = "POTHOLES: ACTIVE" if potholes else "POTHOLES: SCANNING"
    cv2.putText(img, f"STEREO: ACTIVE  |  TRACKER: ACTIVE  |  {pothole_status}",
                (w - int(390 * s), int(24 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36 * s, (0, 255, 180), max(1, int(1 * s)), cv2.LINE_AA)

    # Telemetry line
    v_cnt = density_telemetry.get("visible_vehicle_count", 0)
    t_cnt = density_telemetry.get("tracked_vehicle_count", 0)
    p_cnt = len(potholes)

    ego = traffic_scene.get("ego_metrics", {})
    nearest_dist = ego.get("nearest_vehicle_distance_m")
    nearest_cls = ego.get("nearest_vehicle_class", "")
    nearest_gap = ego.get("nearest_longitudinal_gap_m")

    near_str = f"{nearest_dist:.1f}m ({nearest_cls})" if nearest_dist is not None else "NONE"
    gap_str = f"{nearest_gap:.1f}m" if nearest_gap is not None else "N/A"

    dens_val = density_telemetry.get("traffic_density_veh_km2")
    dens_status = density_telemetry.get("density_status", "DENSITY: UNRELIABLE")
    if dens_status == "OK" and dens_val is not None:
        dens_str = f"{dens_val:.0f} veh/km²"
        dens_color = (0, 255, 180)
    else:
        dens_str = "UNRELIABLE"
        dens_color = (80, 100, 255)

    telem_str = f"VEHICLES: {v_cnt} (Tracked: {t_cnt})  |  NEAREST: {near_str}  |  GAP [HEURISTIC]: {gap_str}  |  POTHOLES: {p_cnt}  |  DENSITY: "
    cv2.putText(img, telem_str, (int(15 * s), int(48 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43 * s, (200, 220, 240), max(1, int(1 * s)), cv2.LINE_AA)
    
    # Render density value with color highlight
    (tw, _), _ = cv2.getTextSize(telem_str, cv2.FONT_HERSHEY_SIMPLEX, 0.43 * s, 1)
    cv2.putText(img, dens_str, (int(15 * s) + tw, int(48 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.44 * s, dens_color, max(1, int(1.2 * s)), cv2.LINE_AA)

    # Bottom timestamp & source info bar
    bot_h = int(24 * s)
    h_img = img.shape[0]
    bot_sub = img[h_img - bot_h:h_img, 0:w]
    bot_bg = np.full_like(bot_sub, (12, 14, 18))
    img[h_img - bot_h:h_img, 0:w] = cv2.addWeighted(bot_bg, 0.85, bot_sub, 0.15, 0)
    
    time_sec = frame_idx / max(1.0, fps)
    info_str = f"Frame: {frame_idx:04d}/{total_frames:04d} | Time: {time_sec:4.1f}s | Source: Forward-Facing Stereo Video (SBS) | Pure Stereo Metrology (Zero GT/Monocular)"
    cv2.putText(img, info_str, (int(15 * s), h_img - int(7 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, (160, 175, 190), max(1, int(1 * s)), cv2.LINE_AA)


def draw_vehicle_card(img: np.ndarray, veh: Dict[str, Any], s: float = 1.0):
    """Draws bounding box, ground-contact point, and floating translucent vehicle card."""
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = veh["bbox"]
    disp_id = veh.get("display_id", f"Vehicle #{veh.get('track_id', 0)}")
    cls_name = veh.get("class", "vehicle").upper()
    conf = veh.get("confidence", 1.0)
    dist_m = veh.get("distance_m")
    status = veh.get("status", "OK")
    method = veh.get("position_method", "stereo")
    X_m = veh.get("X_m")
    Y_m = veh.get("Y_m")
    Z_m = veh.get("Z_m")

    # 1. Bounding box in vibrant emerald green
    box_color = (0, 230, 100)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, max(1, int(1.8 * s)))

    # 2. Ground-contact bottom-center reference point (Fix 2)
    u_bc = int((x1 + x2) / 2)
    v_bc = int(y2)
    cv2.circle(img, (u_bc, v_bc), max(3, int(4 * s)), (0, 255, 255), -1)
    cv2.circle(img, (u_bc, v_bc), max(5, int(6 * s)), (0, 180, 255), 1)

    # 3. Position Floating Card
    card_w = int(220 * s)
    card_h = int(82 * s)
    card_x = max(10, min(x1, w_img - card_w - 10))
    card_y = y1 - card_h - int(8 * s)
    if card_y < int(75 * s):  # Flip below if near banner
        card_y = y2 + int(8 * s)
        if card_y + card_h > h_img - int(30 * s):
            card_y = max(int(75 * s), h_img - card_h - int(30 * s))

    # Translucent card background
    card_roi = img[card_y:card_y + card_h, card_x:card_x + card_w]
    if card_roi.shape[0] == card_h and card_roi.shape[1] == card_w:
        bg = np.full_like(card_roi, (20, 26, 32))
        img[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(bg, 0.85, card_roi, 0.15, 0)
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), box_color, 1)

        # Leader line to vehicle ground-contact
        cv2.line(img, (card_x + card_w // 2, card_y + card_h if card_y < y1 else card_y),
                 (u_bc, y1 if card_y < y1 else y2), (0, 200, 100), max(1, int(1 * s)), cv2.LINE_AA)

        # Card Text
        header = f"{disp_id} [{cls_name}]"
        cv2.putText(img, header, (card_x + int(8 * s), card_y + int(20 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44 * s, (255, 255, 255), max(1, int(1.2 * s)), cv2.LINE_AA)

        if status == "OK" and dist_m is not None:
            pos_str = f"3D: ({X_m:+.1f}, {Y_m:+.1f}, {Z_m:.1f})m"
            cv2.putText(img, f"Dist : {dist_m:.1f} m", (card_x + int(8 * s), card_y + int(40 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (0, 255, 180), max(1, int(1 * s)), cv2.LINE_AA)
            cv2.putText(img, pos_str, (card_x + int(8 * s), card_y + int(58 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, (200, 220, 240), max(1, int(1 * s)), cv2.LINE_AA)
            m_label = "Ray-Plane Contact" if method == "ray_road_plane_intersection" else "Stereo Contact"
            cv2.putText(img, f"Ref: {m_label}", (card_x + int(8 * s), card_y + int(74 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32 * s, (150, 170, 190), max(1, int(1 * s)), cv2.LINE_AA)
        else:
            cv2.putText(img, "DISTANCE: UNRELIABLE", (card_x + int(8 * s), card_y + int(45 * s)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.40 * s, (80, 100, 255), max(1, int(1 * s)), cv2.LINE_AA)
            cv2.putText(img, "[Disparity check failed]", (card_x + int(8 * s), card_y + int(68 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34 * s, (140, 150, 180), max(1, int(1 * s)), cv2.LINE_AA)


def draw_pothole_card(img: np.ndarray, pot: Dict[str, Any], s: float = 1.0):
    """Draws segmented pothole mask and floating translucent pothole card."""
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = pot["bbox"]
    t_id = pot["track_id"]
    conf = pot["confidence"]
    depth_cm = pot.get("depth_p95_cm")
    area_m2 = pot.get("area_m2")
    len_m = pot.get("length_m")
    wid_m = pot.get("width_m")
    quality = pot.get("quality", "GOOD")

    # STRICT RULE: Suppress visual overlay completely if measurement is invalid.
    if quality == "INVALID" or depth_cm is None or area_m2 is None or depth_cm <= 0 or area_m2 <= 0:
        return

    # 1. Mask overlay strictly on pothole
    mask = pot.get("mask")
    if mask is not None and np.any(mask > 0):
        roi = (mask > 0)
        cyan = np.array([255, 180, 0], dtype=np.uint8)  # Cyan in BGR
        img[roi] = (img[roi].astype(float) * 0.50 + cyan.astype(float) * 0.50).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 240, 255), max(2, int(2 * s)), cv2.LINE_AA)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), max(1, int(1.5 * s)))

    # 2. Position card
    card_w = int(220 * s)
    card_h = int(85 * s)
    card_x = max(10, min(x1, w_img - card_w - 10))
    card_y = y1 - card_h - int(8 * s)
    if card_y < int(75 * s):
        card_y = y2 + int(8 * s)
        if card_y + card_h > h_img - int(30 * s):
            card_y = max(int(75 * s), h_img - card_h - int(30 * s))

    card_roi = img[card_y:card_y + card_h, card_x:card_x + card_w]
    if card_roi.shape[0] == card_h and card_roi.shape[1] == card_w:
        bg = np.full_like(card_roi, (28, 24, 20))
        img[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(bg, 0.85, card_roi, 0.15, 0)
        border_col = (0, 255, 128) if quality == "GOOD" else (0, 190, 255) if quality == "WARNING" else (50, 50, 240)
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), border_col, 1)

        # Leader line
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.line(img, (card_x + card_w // 2, card_y + card_h if card_y < y1 else card_y),
                 (center_x, y1 if card_y < y1 else y2), border_col, max(1, int(1 * s)), cv2.LINE_AA)

        header = f"POTHOLE #{t_id:02d} ({conf*100:.0f}%)"
        cv2.putText(img, header, (card_x + int(8 * s), card_y + int(20 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44 * s, (255, 255, 255), max(1, int(1.2 * s)), cv2.LINE_AA)

        if quality != "INVALID" and depth_cm is not None and area_m2 is not None:
            cv2.putText(img, f"Depth : {depth_cm:.1f} cm", (card_x + int(8 * s), card_y + int(40 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (0, 255, 128), max(1, int(1 * s)), cv2.LINE_AA)
            cv2.putText(img, f"Area  : {area_m2:.3f} m²", (card_x + int(8 * s), card_y + int(60 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (0, 215, 255), max(1, int(1 * s)), cv2.LINE_AA)
            if len_m is not None and wid_m is not None:
                cv2.putText(img, f"Dims  : {len_m:.2f}m x {wid_m:.2f}m", (card_x + int(8 * s), card_y + int(78 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34 * s, (200, 200, 200), max(1, int(1 * s)), cv2.LINE_AA)
        else:
            pass


def colorize_disparity(disp: np.ndarray, max_disp: float = 64.0) -> np.ndarray:
    """Colorizes disparity using the Plasma colormap with normalization."""
    valid = (disp > 0.5) & np.isfinite(disp)
    disp_norm = np.zeros_like(disp, dtype=np.uint8)
    if np.any(valid):
        d_val = np.clip(disp, 0.0, max_disp) / max_disp * 255.0
        disp_norm = d_val.astype(np.uint8)
    color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_PLASMA)
    color[~valid] = (20, 20, 20)
    return color


def colorize_depth(depth_m: np.ndarray, max_d: float = 35.0) -> np.ndarray:
    """Colorizes metric depth using the Viridis colormap."""
    valid = (depth_m >= 0.5) & (depth_m <= max_d) & np.isfinite(depth_m)
    depth_norm = np.zeros_like(depth_m, dtype=np.uint8)
    if np.any(valid):
        d_val = (1.0 - (np.clip(depth_m, 0.5, max_d) - 0.5) / (max_d - 0.5)) * 255.0
        depth_norm = d_val.astype(np.uint8)
    color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
    color[~valid] = (15, 15, 15)
    return color


def generate_videos(
    video_source: Path,
    target_demo_path: Path,
    target_debug_path: Path,
    max_duration_s: float = 23.0,
    stride: int = 3,
    target_fps: int = 20
):
    print("\n=======================================================")
    print(" BUSSENSE COMBINED DEMO & DEBUG VIDEO GENERATION")
    print("=======================================================")
    print(f"Source Video:   {video_source}")
    print(f"Demo Target:    {target_demo_path}")
    print(f"Debug Target:   {target_debug_path}")

    cap = cv2.VideoCapture(str(video_source))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_w = full_w // 2
    cam_h = full_h

    print(f"Source Resolution: {full_w}x{full_h} (Each Eye: {cam_w}x{cam_h}) @ {orig_fps:.0f} FPS")

    # Limit frames to target duration
    max_src_frames = int(max_duration_s * orig_fps)
    frame_limit = min(total_src_frames, max_src_frames)
    target_frame_indices = list(range(0, frame_limit, stride))
    num_out_frames = len(target_frame_indices)
    print(f"Processing {num_out_frames} frames (stride={stride}) -> Duration: {num_out_frames / target_fps:.1f}s @ {target_fps} FPS\n")

    # Camera calibration (standard 70° FOV, 12cm stereo baseline)
    fx = 457.0
    fy = 457.0
    cx = cam_w / 2.0
    cy = cam_h / 2.0
    baseline = 0.12
    calib = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline}

    # Initialize Perception Pipeline
    pipeline = BusSensePerceptionPipeline(calib=calib, stereo_model="sgbm")

    # Upscale render output for crisp HD display (1280x720)
    out_w, out_h = 1280, 720
    scale = out_w / float(cam_w)

    # Initialize Video Writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    demo_writer = cv2.VideoWriter(str(target_demo_path), fourcc, target_fps, (out_w, out_h))
    debug_writer = cv2.VideoWriter(str(target_debug_path), fourcc, target_fps, (out_w, out_h))

    t_start = time.time()
    for out_idx, src_idx in enumerate(target_frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
        ret, sbs_frame = cap.read()
        if not ret or sbs_frame is None:
            break

        left_rgb = sbs_frame[:, :cam_w]
        right_rgb = sbs_frame[:, cam_w:]

        # Run Combined Perception Pipeline
        out = pipeline.process_frame(left_rgb, right_rgb, frame_index=out_idx)

        # -------------------------------------------------------------
        # 1. RENDER PRODUCTION DEMO (Full HD Screen)
        # -------------------------------------------------------------
        demo_frame = cv2.resize(left_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Scale detections to output resolution
        scaled_potholes = []
        for p in out["potholes"]:
            sp = dict(p)
            b = p["bbox"]
            sp["bbox"] = [int(b[0] * scale), int(b[1] * scale), int(b[2] * scale), int(b[3] * scale)]
            if p.get("mask") is not None:
                sp["mask"] = cv2.resize(p["mask"].astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            scaled_potholes.append(sp)

        scaled_vehicles = []
        for v in out["traffic"]["vehicles"]:
            sv = dict(v)
            b = v["bbox"]
            sv["bbox"] = [int(b[0] * scale), int(b[1] * scale), int(b[2] * scale), int(b[3] * scale)]
            scaled_vehicles.append(sv)

        # Draw Pothole Masks & Cards
        for sp in scaled_potholes:
            draw_pothole_card(demo_frame, sp, s=scale)

        # Draw Vehicle Boxes, Ground Contacts, and Cards
        for sv in scaled_vehicles:
            draw_vehicle_card(demo_frame, sv, s=scale)

        # Draw Global Telemetry HUD Banner
        draw_combined_hud_banner(
            img=demo_frame,
            frame_idx=out_idx + 1,
            total_frames=num_out_frames,
            fps=float(target_fps),
            traffic_scene=out["traffic"],
            density_telemetry=out["density"],
            potholes=scaled_potholes,
            s=scale
        )

        demo_writer.write(demo_frame)

        # -------------------------------------------------------------
        # 2. RENDER 2x2 SYNCHRONIZED DEBUG QUAD
        # -------------------------------------------------------------
        # Quad resolution: each tile is out_w // 2, out_h // 2 (640x360)
        qw, qh = out_w // 2, out_h // 2

        # Q1 (Top-Left): Left RGB + Clean Overlays
        q1 = left_rgb.copy()
        for p in out["potholes"]:
            bx1, by1, bx2, by2 = p["bbox"]
            cv2.rectangle(q1, (bx1, by1), (bx2, by2), (0, 215, 255), 2)
            if p.get("mask") is not None and np.any(p["mask"] > 0):
                q1[(p["mask"] > 0)] = (q1[(p["mask"] > 0)].astype(float) * 0.55 + np.array([255, 180, 0]) * 0.45).astype(np.uint8)
        for v in out["traffic"]["vehicles"]:
            bx1, by1, bx2, by2 = v["bbox"]
            cv2.rectangle(q1, (bx1, by1), (bx2, by2), (0, 230, 100), 2)
            v_id = v.get("display_id", "Vehicle")
            v_dist = f"{v.get('distance_m')}m" if v.get("distance_m") is not None else "UNREL"
            cv2.putText(q1, f"{v_id} ({v_dist})", (bx1, max(16, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 100), 1, cv2.LINE_AA)
        q1 = cv2.resize(q1, (qw, qh))
        cv2.putText(q1, "Q1: LEFT RGB + VEHICLES & POTHOLES", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Q2 (Top-Right): Right RGB + Horizontal Epipolar Verification Grid
        q2 = cv2.resize(right_rgb, (qw, qh))
        # Draw epipolar grid lines
        for y_line in range(30, qh, 40):
            cv2.line(q2, (0, y_line), (qw, y_line), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(q2, "Q2: RIGHT RGB + EPIPOLAR VERIFICATION", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Q3 (Bottom-Left): Stereo Disparity (Plasma Colormap)
        disp_color = colorize_disparity(out["disparity"])
        q3 = cv2.resize(disp_color, (qw, qh))
        cv2.putText(q3, "Q3: STEREO DISPARITY MAP (PLASMA)", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

        # Q4 (Bottom-Right): Metric Depth (Viridis Colormap) + Telemetry Overlay
        depth_color = colorize_depth(out["depth_m"])
        q4 = cv2.resize(depth_color, (qw, qh))
        cv2.putText(q4, "Q4: METRIC DEPTH Z=(f*B)/d (VIRIDIS)", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)
        dens_txt = f"Density: {out['density'].get('traffic_density_veh_km2', 'N/A')} veh/km²"
        cv2.putText(q4, dens_txt, (10, qh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Assemble 2x2 grid
        top_row = np.hstack([q1, q2])
        bot_row = np.hstack([q3, q4])
        debug_frame = np.vstack([top_row, bot_row])

        debug_writer.write(debug_frame)

        if (out_idx + 1) % 50 == 0 or (out_idx + 1) == num_out_frames:
            elapsed = time.time() - t_start
            fps_proc = (out_idx + 1) / max(1e-3, elapsed)
            print(f"  Frame [{out_idx+1:03d}/{num_out_frames:03d}] | Processed at {fps_proc:.1f} FPS | Elapsed: {elapsed:.1f}s")

    cap.release()
    demo_writer.release()
    debug_writer.release()

    total_time = time.time() - t_start
    print(f"\nCompleted Video Generation in {total_time:.1f} seconds!")
    print(f"  Final Demo Video:  {target_demo_path} ({target_demo_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Final Debug Video: {target_debug_path} ({target_debug_path.stat().st_size / (1024*1024):.2f} MB)")
    print("=======================================================\n")


if __name__ == "__main__":
    src_vid = PROJECT_ROOT / "temp_candidates" / "main_street_sbs.mp4"
    demo_out = PROJECT_ROOT / "outputs" / "demo" / "bussense_vehicle_pothole_demo.mp4"
    debug_out = PROJECT_ROOT / "outputs" / "demo" / "bussense_vehicle_pothole_debug.mp4"

    generate_videos(
        video_source=src_vid,
        target_demo_path=demo_out,
        target_debug_path=debug_out,
        max_duration_s=23.0,
        stride=3,
        target_fps=20
    )
