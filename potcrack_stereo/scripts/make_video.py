"""
make_video.py - BusSense Vehicle Stereo Pothole 3D Measurement Video Generator

Generates:
1. outputs/demo/pothole_heavy_stereo_demo.mp4
   - High-definition full-screen vehicle road inspection video using REAL pothole-heavy stereo sequence.
   - Highlights detected potholes with alpha blending.
   - Floating Translucent HUD Cards displaying:
     * Pothole ID
     * Estimated Depth in cm (95th percentile signed perpendicular depression below local road plane)
     * Estimated Footprint Area in m² (ray-intersected Shoelace formula)
     * Physical Dimensions: Length x Width (m)
     * Confidence %
     * Quality Badge: GOOD (green), WARNING (amber), or 'Depth/area unreliable' (red).
   - Global Telemetry HUD banner with live FPS, frame counter, and pipeline state.

2. outputs/demo/pothole_heavy_stereo_debug.mp4
   - 2x2 Synchronized Multi-Panel Inspection Video:
     * Top-Left: Left RGB + Segmented Potholes & HUD Cards
     * Top-Right: Right RGB with Epipolar Reference Lines
     * Bottom-Left: Stereo Disparity Colormap (plasma)
     * Bottom-Right: Predicted Depth Map / Road Surface Metrology (viridis)

3. outputs/demo/pothole_heavy_performance.json
   - Performance latency breakdown and summary statistics.
"""

import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.run_pipeline import BusSenseStereoPipeline

DEMO_DIR = PROJECT_ROOT / "outputs" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
RSRD_DIR = PROJECT_ROOT / "data" / "RSRD"
G7_DIR = PROJECT_ROOT / "data" / "PavementDamagesG7" / "extracted" / "IMAGES"


def draw_hud_banner(img: np.ndarray, frame_idx: int, total_frames: int, fps: float, num_potholes: int):
    """
    Renders sleek translucent top telemetry HUD banner scaled to resolution.
    """
    h, w = img.shape[:2]
    s = max(1.0, w / 1280.0)
    banner_h = int(60 * s)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
    cv2.line(img, (0, banner_h), (w, banner_h), (0, 215, 255), max(2, int(2 * s)))

    # Title & Subtitle
    cv2.putText(img, "BUSSENSE VEHICLE STEREO 3D METROLOGY", (int(20 * s), int(26 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.68 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(img, "Stereo Inference Only (No GT / No Monocular) | Fast-FoundationStereo + YOLO11-seg",
                (int(20 * s), int(48 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * s, (180, 200, 210), max(1, int(1 * s)), cv2.LINE_AA)

    # Telemetry metrics on right
    fps_text = f"FPS: {fps:4.1f}"
    frame_text = f"FRAME: {frame_idx+1:02d}/{total_frames:02d}"
    pot_text = f"POTHOLES: {num_potholes}"

    cv2.putText(img, fps_text, (w - int(420 * s), int(36 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.58 * s, (0, 255, 128), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(img, frame_text, (w - int(280 * s), int(36 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.52 * s, (220, 220, 220), max(1, int(1 * s)), cv2.LINE_AA)
    cv2.putText(img, pot_text, (w - int(130 * s), int(36 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.52 * s,
                (0, 215, 255) if num_potholes > 0 else (160, 160, 160), max(1, int(2 * s)), cv2.LINE_AA)


def draw_pothole_card(img: np.ndarray, pothole: dict):
    """
    Renders floating translucent inspection HUD card for a detected pothole.
    """
    h_img, w_img = img.shape[:2]
    s = max(1.0, w_img / 1280.0)

    bbox = pothole["bbox"]
    x1, y1, x2, y2 = bbox
    t_id = pothole["track_id"]
    conf = pothole["confidence"]
    depth_cm = pothole["depth_p95_cm"]
    area_m2 = pothole["area_m2"]
    len_m = pothole["length_m"]
    wid_m = pothole["width_m"]
    quality = pothole["quality"]

    # STRICT RULE: Suppress visual overlay completely if measurement is invalid.
    if quality == "INVALID" or depth_cm is None or area_m2 is None or depth_cm <= 0 or area_m2 <= 0:
        return

    # 1. Overlay colored mask on the pothole
    mask = pothole.get("mask")
    if mask is not None:
        color_mask = np.zeros_like(img)
        # Amber fill
        color_mask[mask > 0] = (0, 165, 255) # BGR Amber
        cv2.addWeighted(color_mask, 0.35, img, 0.65, 0, img)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 215, 255), max(2, int(2 * s)), cv2.LINE_AA)

    # 2. Draw bounding box with corner brackets
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), max(1, int(1.5 * s)))

    # 3. Position HUD Card above or to the side of the pothole
    card_w = int(240 * s)
    card_h = int(105 * s)
    card_x = max(10, min(x1, w_img - card_w - 10))
    card_y = y1 - card_h - int(12 * s)
    if card_y < int(70 * s): # Flip below if near top
        card_y = y2 + int(12 * s)
        if card_y + card_h > h_img - 10:
            card_y = h_img - card_h - 10

    # Draw card background
    overlay = img.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 30, 38), -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

    # Border color based on quality
    if quality == "GOOD":
        border_col = (0, 255, 128) # Green
    elif quality == "WARNING":
        border_col = (0, 190, 255) # Amber
    else:
        border_col = (50, 50, 240) # Red
    cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), border_col, max(2, int(2 * s)))

    # Connect leader line from card to pothole center
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    card_anchor = (card_x + card_w // 2, card_y + card_h if card_y < y1 else card_y)
    cv2.line(img, card_anchor, (center_x, y1 if card_y < y1 else y2), border_col, max(1, int(1.5 * s)), cv2.LINE_AA)
    cv2.circle(img, (center_x, center_y), int(4 * s), (0, 215, 255), -1)

    # Card Content
    header_text = f"POTHOLE #{t_id}  ({conf*100:.0f}%)"
    cv2.putText(img, header_text, (card_x + int(10 * s), card_y + int(24 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.50 * s, (255, 255, 255), max(1, int(1.5 * s)), cv2.LINE_AA)

    if quality != "INVALID":
        d_str = f"Depth : {depth_cm:.1f} cm" if depth_cm is not None else "Depth : N/A"
        a_str = f"Area  : {area_m2:.3f} m2" if area_m2 is not None else "Area  : N/A"
        sz_str = f"Dims  : {len_m:.2f}m x {wid_m:.2f}m" if len_m is not None and wid_m is not None else ""

        cv2.putText(img, d_str, (card_x + int(10 * s), card_y + int(48 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46 * s, (0, 255, 128), max(1, int(1.5 * s)), cv2.LINE_AA)
        cv2.putText(img, a_str, (card_x + int(10 * s), card_y + int(72 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46 * s, (0, 215, 255), max(1, int(1.5 * s)), cv2.LINE_AA)
        if sz_str:
            cv2.putText(img, sz_str, (card_x + int(10 * s), card_y + int(94 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, (200, 200, 200), max(1, int(1 * s)), cv2.LINE_AA)


def generate_pothole_heavy_demo(
    sequence_prefix: str = "img_16-10",
    stereo_backend: str = "fast_foundation",
    output_demo_name: str = "pothole_heavy_stereo_demo_15s.mp4",
    output_debug_name: str = "pothole_heavy_stereo_debug_15s.mp4",
    target_duration_s: float = 15.0,
    fps_override: Optional[float] = None,
):
    print("============================================================")
    print("BUSSENSE: GENERATING CONTINUOUS POTHOLE-HEAVY STEREO DEMO VIDEOS")
    print("============================================================")

    # 1. Discover stereo pair files
    left_files = sorted(list(G7_DIR.glob(f"{sequence_prefix}*_i.jpg")))
    if not left_files:
        raise FileNotFoundError(f"No stereo files matching {sequence_prefix} found in {G7_DIR}")

    print(f"Selected Stereo Sequence : {G7_DIR} / {sequence_prefix}*")
    print(f"Total Stereo Pairs       : {len(left_files)}")

    # Calculate playback FPS to achieve target video duration (~15s)
    if fps_override is not None and fps_override > 0:
        fps_target = float(fps_override)
    else:
        fps_target = round(len(left_files) / float(target_duration_s), 3)

    actual_duration = round(len(left_files) / fps_target, 2)
    print(f"Target Duration          : {target_duration_s:.1f}s")
    print(f"Video Playback FPS       : {fps_target:.3f} FPS")
    print(f"Actual Video Duration    : {actual_duration:.2f}s")

    # StereoLabs ZED 2i calibrated intrinsics for 1080p
    calib_zed = {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 960.0,
        "cy": 540.0,
        "baseline": 0.12 # 120 mm
    }

    pipeline = BusSenseStereoPipeline(stereo_backend=stereo_backend)

    first_left = cv2.imread(str(left_files[0]))
    h_orig, w_orig = first_left.shape[:2]

    # Setup Video Writers
    main_video_path = DEMO_DIR / output_demo_name
    debug_video_path = DEMO_DIR / output_debug_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_main = cv2.VideoWriter(str(main_video_path), fourcc, fps_target, (w_orig, h_orig))
    writer_debug = cv2.VideoWriter(str(debug_video_path), fourcc, fps_target, (w_orig, h_orig))

    print(f"Main Demo Output Video   : {main_video_path} ({w_orig}x{h_orig} @ {fps_target:.2f} fps)")
    print(f"Debug 2x2 Output Video   : {debug_video_path} (Synchronized Quad View)")

    latencies_stereo = []
    latencies_seg = []
    latencies_geom = []
    latencies_track = []
    latencies_render = []
    latencies_total = []

    total_detections_count = 0
    count_good = 0
    count_warning = 0
    count_invalid = 0
    seen_track_ids = set()

    for idx, lf in enumerate(left_files):
        stem = lf.stem[:-2] # Remove _i
        rf = G7_DIR / f"{stem}_d.jpg"

        left_img = cv2.imread(str(lf))
        right_img = cv2.imread(str(rf))
        if left_img is None or right_img is None:
            continue

        # Pipeline inference
        res = pipeline.process_frame(left_img, right_img, calib_zed)
        t_pipe = res["timings"]

        # Track stats
        active_potholes = res["tracked_potholes"]
        total_detections_count += len(active_potholes)
        for pot in active_potholes:
            seen_track_ids.add(pot["track_id"])
            q = pot["quality"]
            if q == "GOOD":
                count_good += 1
            elif q == "WARNING":
                count_warning += 1
            else:
                count_invalid += 1

        # Render Main Full-Screen Video
        t_rend_0 = time.time()
        main_frame = left_img.copy()

        for pot in active_potholes:
            draw_pothole_card(main_frame, pot)

        draw_hud_banner(main_frame, idx, len(left_files), t_pipe["fps"], len(active_potholes))
        writer_main.write(main_frame)

        # Render 2x2 Debug Inspection Video
        half_w, half_h = w_orig // 2, h_orig // 2
        q1 = cv2.resize(main_frame, (half_w, half_h))
        cv2.putText(q1, "Q1: LEFT RGB + 3D METROLOGY HUD", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        q2_full = right_img.copy()
        for y_line in range(60, h_orig, 90):
            cv2.line(q2_full, (0, y_line), (w_orig, y_line), (0, 180, 0), 1)
        q2 = cv2.resize(q2_full, (half_w, half_h))
        cv2.putText(q2, "Q2: RIGHT RGB + EPIPOLAR REFERENCE", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        pred_disp = res["pred_disparity"]
        disp_valid = pred_disp[pred_disp > 0]
        vmax_disp = float(np.percentile(disp_valid, 98)) if len(disp_valid) > 0 else 64.0
        disp_norm = np.clip(pred_disp / max(vmax_disp, 1.0) * 255.0, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_PLASMA)
        disp_color[pred_disp <= 0] = 0
        q3 = cv2.resize(disp_color, (half_w, half_h))
        cv2.putText(q3, f"Q3: STEREO DISPARITY (MAX: {vmax_disp:.1f} px)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        pred_depth = res["pred_depth"]
        depth_valid = pred_depth[(pred_depth >= 0.2) & (pred_depth <= 20.0)]
        vmax_depth = float(np.percentile(depth_valid, 95)) if len(depth_valid) > 0 else 10.0
        depth_norm = np.clip(pred_depth / max(vmax_depth, 1.0) * 255.0, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_VIRIDIS)
        depth_color[pred_depth <= 0] = 0
        q4 = cv2.resize(depth_color, (half_w, half_h))
        cv2.putText(q4, f"Q4: PREDICTED DEPTH (MAX: {vmax_depth:.1f} m)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        top_row = np.hstack([q1, q2])
        bottom_row = np.hstack([q3, q4])
        debug_frame = np.vstack([top_row, bottom_row])
        cv2.line(debug_frame, (half_w, 0), (half_w, h_orig), (255, 255, 255), 2)
        cv2.line(debug_frame, (0, half_h), (w_orig, half_h), (255, 255, 255), 2)
        writer_debug.write(debug_frame)

        t_rend = (time.time() - t_rend_0) * 1000.0

        latencies_stereo.append(t_pipe["stereo_ms"])
        latencies_seg.append(t_pipe["seg_ms"])
        latencies_geom.append(t_pipe["geometry_ms"])
        latencies_track.append(t_pipe["tracking_ms"])
        latencies_render.append(t_rend)
        latencies_total.append(t_pipe["total_ms"] + t_rend)

        print(f"  Frame [{idx+1:02d}/{len(left_files):02d}] {stem}: "
              f"Potholes={len(active_potholes)} (Tracks={len(seen_track_ids)}) | "
              f"Pipe={t_pipe['total_ms']:.1f}ms | Render={t_rend:.1f}ms")

    writer_main.release()
    writer_debug.release()

    # Performance report
    actual_inf_fps = round(float(1000.0 / np.mean(latencies_total)), 2)
    valid_measurements_count = count_good + count_warning

    perf = {
        "sequence_name": sequence_prefix,
        "stereo_sequence_path": str(G7_DIR / f"{sequence_prefix}*"),
        "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "input_resolution": f"{w_orig}x{h_orig}",
        "stereo_model": pipeline.stereo_matcher.model_name,
        "segmentation_model": "YOLO11n-seg (Fine-tuned on G-7)",
        "frames_processed": len(left_files),
        "target_duration_seconds": target_duration_s,
        "actual_duration_seconds": actual_duration,
        "video_playback_fps": fps_target,
        "inference_fps": actual_inf_fps,
        "total_pothole_detections": total_detections_count,
        "unique_potholes_tracked": len(seen_track_ids),
        "valid_depth_area_measurements": valid_measurements_count,
        "quality_counts": {
            "GOOD": count_good,
            "WARNING": count_warning,
            "INVALID": count_invalid
        },
        "mean_latency_ms": {
            "stereo_disparity": round(float(np.mean(latencies_stereo)), 2),
            "pothole_segmentation": round(float(np.mean(latencies_seg)), 2),
            "local_road_plane_3d_metrology": round(float(np.mean(latencies_geom)), 2),
            "temporal_tracking": round(float(np.mean(latencies_track)), 2),
            "video_hud_rendering": round(float(np.mean(latencies_render)), 2),
            "total_end_to_end": round(float(np.mean(latencies_total)), 2)
        },
        "output_files": {
            "main_demo_mp4": str(main_video_path),
            "debug_quad_mp4": str(debug_video_path)
        }
    }

    perf_file_15s = DEMO_DIR / "pothole_heavy_performance_15s.json"
    with open(perf_file_15s, "w", encoding="utf-8") as pf:
        json.dump(perf, pf, indent=4)

    # Also keep general perf file updated
    with open(DEMO_DIR / "pothole_heavy_performance.json", "w", encoding="utf-8") as pf:
        json.dump(perf, pf, indent=4)

    print("\n------------------------------------------------------------")
    print("POTHOLE-HEAVY STEREO 15S VIDEO GENERATION COMPLETE:")
    print("------------------------------------------------------------")
    print(f"Selected Sequence             : {sequence_prefix} ({len(left_files)} frames)")
    print(f"Actual Video Duration         : {actual_duration:.2f} seconds")
    print(f"Number of Frames Processed    : {len(left_files)}")
    print(f"Video Playback FPS            : {fps_target:.3f} FPS")
    print(f"Inference Processing FPS      : {actual_inf_fps:.2f} FPS")
    print(f"Unique Potholes Tracked       : {len(seen_track_ids)}")
    print(f"Total Pothole Detections      : {total_detections_count}")
    print(f"Valid Depth/Area Measurements : {valid_measurements_count} (GOOD: {count_good}, WARN: {count_warning})")
    print(f"Invalid Measurements          : {count_invalid}")
    print(f"Main Demo Video Path          : {main_video_path} ({main_video_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Debug 2x2 Video Path          : {debug_video_path} ({debug_video_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Performance JSON              : {perf_file_15s}")
    print("============================================================\n")

    return perf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="img_16-10", help="Sequence prefix to process")
    parser.add_argument("--stereo_backend", type=str, default="fast_foundation", help="Stereo model backend")
    parser.add_argument("--demo_name", type=str, default="pothole_heavy_stereo_demo_15s.mp4", help="Main demo filename")
    parser.add_argument("--debug_name", type=str, default="pothole_heavy_stereo_debug_15s.mp4", help="Debug demo filename")
    parser.add_argument("--duration", type=float, default=15.0, help="Target video duration in seconds")
    parser.add_argument("--fps", type=float, default=None, help="Explicit playback FPS override")
    args = parser.parse_args()

    generate_pothole_heavy_demo(
        sequence_prefix=args.sequence,
        stereo_backend=args.stereo_backend,
        output_demo_name=args.demo_name,
        output_debug_name=args.debug_name,
        target_duration_s=args.duration,
        fps_override=args.fps
    )
