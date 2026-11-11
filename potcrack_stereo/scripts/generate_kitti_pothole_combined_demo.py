#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — KITTI + Vehicular Stereo Dashcam Pothole Unified Test Generator
==========================================================================
Combines:
1. KITTI moving-vehicle stereo dataset (active traffic sensing, ByteTrack, 3D inter-vehicle distance)
2. Real vehicular stereo dashcam video with visible severe potholes (pothole segmentation, 3D depression depth, Shoelace area, crack geometry)
3. Dynamic, evidence-based Road Quality Engine demonstrating live transitions from GOOD/FAIR to CRITICAL/POOR.

Outputs:
- outputs/demo/bussense_kitti_pothole_unified_demo.mp4
- outputs/demo/bussense_kitti_pothole_unified_debug.mp4
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

TARGET_DEMO = DEMO_DIR / "bussense_full_fixed_demo.mp4"
TARGET_DEBUG = DEMO_DIR / "bussense_full_fixed_debug.mp4"

# Standard target HD canvas resolution (widescreen 16:9 or 2.39:1)
CANVAS_W, CANVAS_H = 1280, 540
TARGET_FPS = 10


def resize_letterbox(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scales and letterboxes image to fit target canvas while preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(target_w / float(w), target_h / float(h))
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def draw_unified_pothole_card(img: np.ndarray, pothole: Dict[str, Any], scale: float = 1.0):
    """Draws floating translucent inspection card for detected pothole."""
    h_img, w_img = img.shape[:2]
    s = scale * max(0.8, w_img / 1280.0)
    
    bbox = pothole.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = bbox
    t_id = pothole.get("track_id", 1)
    conf = pothole.get("confidence", 0.8)
    depth_cm = pothole.get("depth_p95_cm")
    area_m2 = pothole.get("area_m2")
    quality = pothole.get("quality", "GOOD")
    
    # STRICT POTHOLE VISUAL RULE:
    # If physical measurement is invalid, suppress visual overlay completely.
    # Track remains alive internally for persistence and analytics.
    if quality == "INVALID" or depth_cm is None or area_m2 is None or depth_cm <= 0 or area_m2 <= 0:
        return
    
    # Overlay pothole mask if present
    mask = pothole.get("mask")
    if mask is not None and np.any(mask > 0):
        color = (0, 69, 255) if quality != "GOOD" else (0, 140, 255)
        overlay = img.copy()
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 215, 255), max(1, int(1.5 * s)), cv2.LINE_AA)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), max(1, int(1.5 * s)))
    
    card_w = int(240 * s)
    card_h = int(92 * s)
    card_x = max(10, min(x1, w_img - card_w - 10))
    card_y = y1 - card_h - int(8 * s)
    if card_y < int(75 * s):
        card_y = y2 + int(8 * s)
        if card_y + card_h > h_img - int(30 * s):
            card_y = max(int(75 * s), h_img - card_h - int(30 * s))
            
    card_roi = img[card_y:card_y + card_h, card_x:card_x + card_w]
    if card_roi.shape[0] == card_h and card_roi.shape[1] == card_w:
        bg = np.full_like(card_roi, (28, 22, 18))
        img[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(bg, 0.85, card_roi, 0.15, 0)
        border_col = (0, 255, 128) if quality == "GOOD" else (0, 165, 255) if quality == "WARNING" else (50, 50, 240)
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), border_col, 1)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.line(img, (card_x + card_w // 2, card_y + card_h if card_y < y1 else card_y),
                 (center_x, y1 if card_y < y1 else y2), border_col, 1, cv2.LINE_AA)

        header = f"POTHOLE #{t_id:02d} ({conf*100:.0f}%)"
        cv2.putText(img, header, (card_x + int(8 * s), card_y + int(20 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44 * s, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(img, f"Depth : {depth_cm:.1f} cm", (card_x + int(8 * s), card_y + int(44 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (0, 255, 128), 1, cv2.LINE_AA)
        cv2.putText(img, f"Area  : {area_m2:.3f} m2", (card_x + int(8 * s), card_y + int(68 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (0, 215, 255), 1, cv2.LINE_AA)


def draw_unified_vehicle_card(img: np.ndarray, vehicle: Dict[str, Any], scale: float = 1.0):
    """Draws floating translucent 3D tracking card for tracked vehicle."""
    w_img = img.shape[1]
    s = scale * max(0.8, w_img / 1280.0)
    
    bbox = vehicle["bbox"]
    x1, y1, x2, y2 = bbox
    disp_id = vehicle.get("display_id", f"Vehicle #{vehicle.get('track_id', 1):02d}")
    v_class = vehicle.get("class", "CAR").upper()
    dist_m = vehicle.get("distance_m")
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 130), max(1, int(1.5 * s)))
    
    # Ground contact circle
    gc_px = vehicle.get("ground_contact_px")
    if gc_px:
        cv2.circle(img, (int(gc_px[0]), int(gc_px[1])), max(3, int(4 * s)), (0, 255, 255), -1)
        
    tag_w = int(170 * s)
    tag_h = int(38 * s)
    tag_x = max(5, min(x1, w_img - tag_w - 5))
    tag_y = max(5, y1 - tag_h - int(4 * s))
    
    tag_roi = img[tag_y:tag_y + tag_h, tag_x:tag_x + tag_w]
    if tag_roi.shape[0] == tag_h and tag_roi.shape[1] == tag_w:
        bg = np.full_like(tag_roi, (15, 24, 20))
        img[tag_y:tag_y + tag_h, tag_x:tag_x + tag_w] = cv2.addWeighted(bg, 0.85, tag_roi, 0.15, 0)
        cv2.rectangle(img, (tag_x, tag_y), (tag_x + tag_w, tag_y + tag_h), (0, 220, 130), 1)
        
        cv2.putText(img, f"{disp_id} | {v_class}", (tag_x + int(6 * s), tag_y + int(16 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38 * s, (255, 255, 255), 1, cv2.LINE_AA)
        d_str = f"Dist: {dist_m:.1f}m" if dist_m is not None else "Dist: N/A"
        cv2.putText(img, d_str, (tag_x + int(6 * s), tag_y + int(32 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36 * s, (0, 240, 255), 1, cv2.LINE_AA)


def draw_unified_crack(img: np.ndarray, crack: Dict[str, Any], scale: float = 1.0):
    """Draws crack segmentation overlay and annotation box."""
    w_img = img.shape[1]
    s = scale * max(0.8, w_img / 1280.0)

    mask = crack.get("mask")
    if mask is not None and np.any(mask > 0):
        overlay = img.copy()
        overlay[mask > 0] = (255, 0, 180)  # vibrant magenta for cracks
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 120, 255), max(1, int(1.2 * s)), cv2.LINE_AA)

    bbox = crack.get("bbox")
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (220, 50, 220), max(1, int(1.2 * s)))
        c_len = crack.get("length_m")
        t_id = crack.get("track_id", 1)
        lbl = f"CRACK #{t_id}" + (f" ({c_len:.1f}m)" if c_len is not None else "")
        cv2.putText(img, lbl, (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36 * s, (255, 160, 255), 1, cv2.LINE_AA)


def draw_comprehensive_hud_banner(
    img: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    source_name: str,
    state: UnifiedFrameState
):
    """Draws multi-tier top telemetry HUD banner with dynamic Road Quality readout."""
    h, w = img.shape[:2]
    s = max(0.8, w / 1280.0)
    banner_h = int(82 * s)

    sub = img[0:banner_h, 0:w]
    bg = np.full_like(sub, (16, 20, 26))
    img[0:banner_h, 0:w] = cv2.addWeighted(bg, 0.88, sub, 0.12, 0)
    cv2.line(img, (0, banner_h), (w, banner_h), (0, 200, 255), max(1, int(1.5 * s)))

    # Title Line
    cv2.putText(img, "BUSSENSE UNIFIED VEHICULAR STEREO PERCEPTION", (int(15 * s), int(22 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.44 * s, (255, 255, 255), 1, cv2.LINE_AA)

    # Source Badge
    cv2.putText(img, f"SOURCE: {source_name}", (int(380 * s), int(22 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, (0, 215, 255), 1, cv2.LINE_AA)

    # Frame & FPS (Explicit separation of output video rate vs measured inference speed)
    cv2.putText(img, f"FRAME: {frame_idx+1:03d}/{total_frames:03d}  |  VIDEO: {TARGET_FPS} FPS  |  INFERENCE: {fps:.1f} FPS",
                (w - int(340 * s), int(22 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, (200, 200, 200), 1, cv2.LINE_AA)

    # Telemetry Row 2: Traffic & Distress
    v_cnt = len(state.vehicles)
    p_cnt = len(state.potholes)
    cr_cnt = len(state.cracks)
    ego = state.traffic_metrics or {}
    near_d = ego.get("nearest_vehicle_distance_m")
    near_str = f"{near_d:.1f}m" if near_d is not None else "NONE"
    dens_val = ego.get("traffic_density_veh_km2")
    dens_str = f"{dens_val:.0f} veh/km2" if dens_val is not None else "N/A"
    traf_lvl = ego.get("traffic_level", "NORMAL")

    cv2.putText(img, f"TRAFFIC: {traf_lvl}  |  VEHICLES: {v_cnt}  |  DENSITY: {dens_str}  |  GAP: {near_str}",
                (int(15 * s), int(46 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, (180, 230, 255), 1, cv2.LINE_AA)

    cv2.putText(img, f"POTHOLES: {p_cnt}  |  CRACKS: {cr_cnt}",
                (int(660 * s), int(46 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s,
                (0, 215, 255) if (p_cnt > 0 or cr_cnt > 0) else (180, 180, 180), 1, cv2.LINE_AA)

    # Row 3: ROAD CONDITION READOUT
    rq = state.road_quality or {}
    category = rq.get("condition_category", "NORMAL")
    rms = rq.get("stereo_surface_irregularity_rms_cm")
    conf = rq.get("confidence_percent", 0.0)
    rms_str = f"{rms:.1f} cm" if rms is not None else "N/A"

    # Dynamic color code for operational categories
    if category == "NORMAL":
        cat_color = (0, 230, 115)      # Fresh Green
    elif category == "MEDIUM":
        cat_color = (0, 215, 255)      # Amber / Yellow
    elif category == "HIGH":
        cat_color = (0, 140, 255)      # Orange
    elif category == "ALARMING":
        cat_color = (30, 30, 245)      # Bright Red
    else:
        cat_color = (160, 160, 160)    # Unreliable / Grey

    # Draw Road Condition pill
    pill_x = int(15 * s)
    pill_y = int(58 * s)
    pill_w = int(240 * s)
    pill_h = int(20 * s)
    cv2.rectangle(img, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), cat_color, -1)
    text_color = (15, 15, 20) if category != "ALARMING" else (255, 255, 255)
    cv2.putText(img, f"ROAD CONDITION: {category}", (pill_x + int(8 * s), pill_y + int(14 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.36 * s, text_color, 1, cv2.LINE_AA)

    # Additional road metrics
    cv2.putText(img, f"Surface Residual RMS: {rms_str}  [NOT IRI]  |  Confidence: {conf:.0f}%",
                (pill_x + pill_w + int(15 * s), pill_y + int(14 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, (240, 240, 240), 1, cv2.LINE_AA)


def generate_kitti_pothole_unified_demo():
    print("============================================================")
    print(" BUSSENSE: KITTI + VEHICULAR POTHOLE DASHCAM UNIFIED TEST")
    print("============================================================")

    # 1. Discover KITTI stereo drives
    kitti_drives = discover_drives(KITTI_ROOT)
    kitti_calib = parse_kitti_calib(CALIB_FILE)

    # Select KITTI frames from Drive 0005 (heavy traffic) and Drive 0013
    kitti_frames: List[Tuple[Path, Path, str, Dict[str, float]]] = []
    
    # Drive 0005: First 75 frames for Phase A
    d05_tuple = next(((d_name, ls, rs) for d_name, ls, rs in kitti_drives if "drive_0005" in d_name), None)
    if d05_tuple:
        d_name, l_files, r_files = d05_tuple
        for lf, rf in zip(l_files[:75], r_files[:75]):
            kitti_frames.append((lf, rf, "KITTI DRIVE 0005 (TRAFFIC SENSING)", kitti_calib))

    # 2. Discover Vehicular Pothole Stereo Dataset
    pothole_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
    pot_left_files = sorted(list((pothole_dir / "left").glob("*.jpg")))
    pot_right_files = sorted(list((pothole_dir / "right").glob("*.jpg")))

    calib_zed = {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 960.0,
        "cy": 540.0,
        "baseline": 0.12,
        "image_width": 1920,
        "image_height": 1080
    }

    pothole_frames: List[Tuple[Path, Path, str, Dict[str, float]]] = []
    for lf, rf in zip(pot_left_files, pot_right_files):
        pothole_frames.append((lf, rf, "VEHICULAR STEREO DASHCAM (POTHOLES & QUALITY)", calib_zed))

    # 3. Add second KITTI segment (Traffic Return: frames 25..100)
    kitti_frames_part2: List[Tuple[Path, Path, str, Dict[str, float]]] = []
    if d05_tuple:
        d_name, l_files, r_files = d05_tuple
        for lf, rf in zip(l_files[25:100], r_files[25:100]):
            kitti_frames_part2.append((lf, rf, "KITTI DRIVE 0005 (TRAFFIC RETURN)", kitti_calib))

    # Combine all sequences into one continuous test reel:
    # Phase A: KITTI Traffic (75 frames, 7.5s) -> Road Condition NORMAL
    # Phase B: Vehicular Dashcam Potholes (74 frames, 7.4s) -> Road Condition HIGH/ALARMING
    # Phase C: KITTI Traffic Part 2 (75 frames, 7.5s) -> Road Condition return to NORMAL
    all_sequence_items = kitti_frames + pothole_frames + kitti_frames_part2
    total_frames = len(all_sequence_items)
    total_duration_s = total_frames / float(TARGET_FPS)

    print(f"Total Combined Frames:     {total_frames}")
    print(f"  Segment 1 (KITTI Traffic):   {len(kitti_frames)} frames (Active Traffic Sensing)")
    print(f"  Segment 2 (Potholes/Cracks): {len(pothole_frames)} frames (Visible Distress & Metrology)")
    print(f"  Segment 3 (Traffic Return):  {len(kitti_frames_part2)} frames (Recovery to Normal)")
    print(f"Playback Frame Rate:       {TARGET_FPS} FPS")
    print(f"Total Video Duration:      {total_duration_s:.1f} seconds")
    print(f"Target Render Canvas:      {CANVAS_W}x{CANVAS_H}\n")

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    demo_writer = cv2.VideoWriter(str(TARGET_DEMO), fourcc, TARGET_FPS, (CANVAS_W, CANVAS_H))
    debug_writer = cv2.VideoWriter(str(TARGET_DEBUG), fourcc, TARGET_FPS, (CANVAS_W, CANVAS_H))

    # Crack model path
    crack_weights = str(PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt")

    # Pipelines per calibration (shares models, independent calib)
    pipeline_kitti = UnifiedBusSensePipeline(calib=kitti_calib, stereo_model="sgbm", crack_weights=crack_weights, conf_crack=0.08)
    pipeline_zed = UnifiedBusSensePipeline(calib=calib_zed, stereo_model="sgbm", crack_weights=crack_weights, conf_crack=0.08)

    t_start = time.time()
    for frame_idx, (lf, rf, src_name, calib) in enumerate(all_sequence_items):
        t_f0 = time.time()
        left_img = cv2.imread(str(lf))
        right_img = cv2.imread(str(rf))
        if left_img is None or right_img is None:
            continue

        # Select pipeline based on source
        pipeline = pipeline_zed if "VEHICULAR" in src_name else pipeline_kitti

        # Process frame
        state = pipeline.process_frame(left_img, right_img, frame_index=frame_idx, timestamp=frame_idx / float(TARGET_FPS))

        # -------------------------------------------------------------
        # 1. RENDER MAIN PRODUCTION DEMO
        # -------------------------------------------------------------
        annotated_left = left_img.copy()

        # Draw crack masks and overlays
        for c in state.cracks:
            draw_unified_crack(annotated_left, c)

        # Draw vehicle tracking boxes and 3D cards
        for v in state.vehicles:
            draw_unified_vehicle_card(annotated_left, v)

        # Draw pothole masks and inspection cards
        for p in state.potholes:
            draw_unified_pothole_card(annotated_left, p)

        # Draw top telemetry HUD banner
        f_fps = 1.0 / max(1e-3, time.time() - t_f0)
        draw_comprehensive_hud_banner(annotated_left, frame_idx, total_frames, f_fps, src_name, state)

        # Letterbox into standardized HD canvas
        demo_canvas = resize_letterbox(annotated_left, CANVAS_W, CANVAS_H)
        demo_writer.write(demo_canvas)

        # -------------------------------------------------------------
        # 2. RENDER 2x2 SYNCHRONIZED DIAGNOSTIC QUAD
        # -------------------------------------------------------------
        qw, qh = CANVAS_W // 2, CANVAS_H // 2

        # Q1: Left RGB + Annotations
        q1 = resize_letterbox(annotated_left, qw, qh)
        cv2.putText(q1, "Q1: LEFT RGB + VEHICLES + POTHOLES + HUD", (10, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # Q2: Right RGB + Epipolar Verification Grid
        right_overlay = right_img.copy()
        for y_line in range(30, right_overlay.shape[0], 45):
            cv2.line(right_overlay, (0, y_line), (right_overlay.shape[1], y_line), (0, 255, 0), 1, cv2.LINE_AA)
        q2 = resize_letterbox(right_overlay, qw, qh)
        cv2.putText(q2, "Q2: RIGHT RGB + EPIPOLAR REFERENCE LINES", (10, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

        # Q3: Stereo Disparity (Plasma Colormap)
        disp_color = colorize_disparity(state.disparity)
        q3 = resize_letterbox(disp_color, qw, qh)
        cv2.putText(q3, "Q3: SHARED STEREO DISPARITY (PLASMA)", (10, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 200, 0), 1, cv2.LINE_AA)

        # Q4: Metric Depth (Viridis Colormap) + Live Quality Readout
        depth_color = colorize_depth(state.depth_m)
        q4 = resize_letterbox(depth_color, qw, qh)
        cv2.putText(q4, "Q4: METRIC DEPTH Z=(f*B)/d (VIRIDIS)", (10, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (100, 255, 100), 1, cv2.LINE_AA)

        # Telemetry badge on Q4
        rq = state.road_quality or {}
        cat = rq.get("condition_category", "N/A")
        rms_cm = rq.get("stereo_surface_irregularity_rms_cm")
        rms_txt = f"{rms_cm:.1f}cm" if rms_cm is not None else "N/A"
        cv2.putText(q4, f"Road Condition: {cat} | Residual RMS: {rms_txt}", (10, qh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Combine quad
        top_half = np.hstack([q1, q2])
        bot_half = np.hstack([q3, q4])
        debug_canvas = np.vstack([top_half, bot_half])
        debug_writer.write(debug_canvas)

        if (frame_idx + 1) % 25 == 0 or (frame_idx + 1) == total_frames:
            elapsed = time.time() - t_start
            fps_overall = (frame_idx + 1) / max(1e-3, elapsed)
            print(f"  Frame [{frame_idx+1:03d}/{total_frames:03d}] | {src_name[:25]}... | Condition: {cat} | Speed: {fps_overall:.1f} FPS")

    demo_writer.release()
    debug_writer.release()

    total_time = time.time() - t_start
    print(f"\n============================================================")
    print(f" COMPLETED COMBINED TEST VIDEO GENERATION IN {total_time:.1f} SECONDS")
    print(f"============================================================")
    print(f"  Main Demo Video:  {TARGET_DEMO} ({TARGET_DEMO.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Debug Quad Video: {TARGET_DEBUG} ({TARGET_DEBUG.stat().st_size / (1024*1024):.2f} MB)")
    print(f"============================================================\n")


if __name__ == "__main__":
    generate_kitti_pothole_unified_demo()
