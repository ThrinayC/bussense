"""
make_long_demo.py - BusSense Vehicle Stereo Pothole 3D Measurement Long Demo Generator
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.run_pipeline import BusSenseStereoPipeline
from scripts.make_video import draw_hud_banner

DEMO_DIR = PROJECT_ROOT / "outputs" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data" / "pothole_long_stereo"


def draw_pothole_card_fixed(img: np.ndarray, pothole: dict):
    """
    Renders floating translucent inspection HUD card for a detected pothole.
    Blends mask strictly on pothole pixels to preserve full road brightness.
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

    # 1. Overlay colored mask on the pothole ONLY (preserve road brightness)
    mask = pothole.get("mask")
    if mask is not None and np.any(mask > 0):
        roi = (mask > 0)
        amber = np.array([0, 165, 255], dtype=np.uint8)
        img[roi] = (img[roi].astype(float) * 0.55 + amber.astype(float) * 0.45).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 215, 255), max(2, int(2 * s)), cv2.LINE_AA)

    # 2. Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), max(1, int(1.5 * s)))

    # 3. Position HUD Card above or to the side of the pothole
    card_w = int(240 * s)
    card_h = int(105 * s)
    card_x = max(10, min(x1, w_img - card_w - 10))
    card_y = y1 - card_h - int(12 * s)
    if card_y < int(70 * s):  # Flip below if near top
        card_y = y2 + int(12 * s)
        if card_y + card_h > h_img - 10:
            card_y = h_img - card_h - 10

    # Draw card background strictly in card ROI
    card_roi = img[card_y:card_y + card_h, card_x:card_x + card_w]
    bg = np.full_like(card_roi, (25, 30, 38))
    img[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(bg, 0.88, card_roi, 0.12, 0)

    # Border color based on quality
    if quality == "GOOD":
        border_col = (0, 255, 128)  # Green
    elif quality == "WARNING":
        border_col = (0, 190, 255)  # Amber
    else:
        border_col = (50, 50, 240)  # Red
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


def draw_panel_badge(img: np.ndarray, text: str, x: int = 20, y: int = 35):
    """Draws a sleek translucent badge behind panel titles."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    sub_img = img[y - th - 8:y + 8, x - 8:x + tw + 8]
    bg = np.full_like(sub_img, (18, 22, 28))
    img[y - th - 8:y + 8, x - 8:x + tw + 8] = cv2.addWeighted(bg, 0.85, sub_img, 0.15, 0)
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 215, 255), 1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)


def generate_long_stereo_demo(
    stereo_backend: str = "fast_foundation",
    output_demo_name: str = "pothole_long_stereo_demo.mp4",
    output_debug_name: str = "pothole_long_stereo_debug.mp4",
    target_duration_s: float = 15.0,
    fps_override: Optional[float] = 5.0,
):
    print("============================================================")
    print("BUSSENSE: GENERATING REAL LONG MULTI-POTHOLE STEREO DEMO")
    print("============================================================")

    # 1. Discover stereo pair files
    left_files = sorted(list((DATA_DIR / "left").glob("frame_*.jpg")))
    if not left_files:
        raise FileNotFoundError(f"No stereo files found in {DATA_DIR / 'left'}")

    with open(DATA_DIR / "sequence_meta.json", "r") as f:
        sequence_meta = json.load(f)

    print(f"Stereo Sequence Source    : {DATA_DIR} (Multi-Session Vehicle Stereo Reel)")
    print(f"Total Stereo Pairs        : {len(left_files)}")

    if fps_override is not None and fps_override > 0:
        fps_target = float(fps_override)
    else:
        fps_target = round(len(left_files) / float(target_duration_s), 3)

    actual_duration = round(len(left_files) / fps_target, 2)
    print(f"Target Duration           : {target_duration_s:.1f}s")
    print(f"Video Playback FPS        : {fps_target:.3f} FPS")
    print(f"Actual Video Duration     : {actual_duration:.2f}s")

    # StereoLabs ZED 2i calibrated intrinsics for 1080p
    calib_zed = {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 960.0,
        "cy": 540.0,
        "baseline": 0.12  # 120 mm
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

    print(f"Main Demo Output Video    : {main_video_path} ({w_orig}x{h_orig} @ {fps_target:.2f} fps)")
    print(f"Debug 2x2 Output Video    : {debug_video_path} (Synchronized Quad View)")

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
    all_measured_depths = []
    all_measured_areas = []

    for idx, lf in enumerate(left_files):
        rf = DATA_DIR / "right" / lf.name
        left_img = cv2.imread(str(lf))
        right_img = cv2.imread(str(rf))
        if left_img is None or right_img is None:
            continue

        meta_info = sequence_meta[idx] if idx < len(sequence_meta) else {}
        session_id = meta_info.get("session", "road_test")

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

            if pot["depth_p95_cm"] is not None and pot["depth_p95_cm"] > 0:
                all_measured_depths.append(pot["depth_p95_cm"])
            if pot["area_m2"] is not None and pot["area_m2"] > 0:
                all_measured_areas.append(pot["area_m2"])

        # -------------------------------------------------------------
        # RENDER 1: MAIN DEMO INSPECTION VIDEO
        # -------------------------------------------------------------
        t_rend_0 = time.time()
        main_frame = left_img.copy()

        # Render pothole annotations & cards
        for pot in active_potholes:
            draw_pothole_card_fixed(main_frame, pot)

        # Draw Global HUD Banner with session badge
        draw_hud_banner(main_frame, idx, len(left_files), t_pipe["fps"], len(active_potholes))
        
        # Add session tag in corner
        cv2.putText(main_frame, f"REEL: {session_id}", (20, h_orig - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)

        writer_main.write(main_frame)

        # -------------------------------------------------------------
        # RENDER 2: 2x2 SYNCHRONIZED DEBUG VIDEO
        # -------------------------------------------------------------
        half_w, half_h = w_orig // 2, h_orig // 2

        # Q1: Left RGB + Annotations
        q1 = cv2.resize(main_frame, (half_w, half_h))
        draw_panel_badge(q1, "Q1: LEFT RGB + 3D METROLOGY HUD", 20, 55)

        # Q2: Right RGB + Epipolar lines
        q2 = cv2.resize(right_img, (half_w, half_h))
        for line_y in [half_h // 4, half_h // 2, 3 * half_h // 4]:
            cv2.line(q2, (0, line_y), (half_w, line_y), (0, 255, 0), 1, cv2.LINE_AA)
        draw_panel_badge(q2, "Q2: RIGHT RGB + EPIPOLAR REFERENCE", 20, 35)

        # Q3: Stereo Disparity Colormap (plasma)
        pred_disp = res["pred_disparity"]
        disp_valid = pred_disp[pred_disp > 0]
        vmax_disp = float(np.percentile(disp_valid, 98)) if len(disp_valid) > 0 else 64.0
        disp_norm = np.clip(pred_disp / max(vmax_disp, 1.0) * 255.0, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_PLASMA)
        disp_color[pred_disp <= 0] = 0
        q3 = cv2.resize(disp_color, (half_w, half_h))
        draw_panel_badge(q3, f"Q3: STEREO DISPARITY (MAX: {vmax_disp:.1f} px)", 20, 35)

        # Q4: Predicted Depth Map / Road Surface Metrology (viridis)
        pred_depth = res["pred_depth"]
        depth_valid = pred_depth[(pred_depth >= 0.2) & (pred_depth <= 20.0)]
        vmax_depth = float(np.percentile(depth_valid, 95)) if len(depth_valid) > 0 else 10.0
        depth_norm = np.clip(pred_depth / max(vmax_depth, 1.0) * 255.0, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_VIRIDIS)
        depth_color[pred_depth <= 0] = 0
        q4 = cv2.resize(depth_color, (half_w, half_h))
        draw_panel_badge(q4, f"Q4: PREDICTED DEPTH (MAX: {vmax_depth:.1f} m)", 20, 35)

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

        if (idx + 1) % 10 == 0 or idx == len(left_files) - 1:
            print(f"  Frame [{idx+1:02d}/{len(left_files):02d}] {lf.stem}: "
                  f"Potholes={len(active_potholes)} (Tracks={len(seen_track_ids)}) | "
                  f"Pipe={t_pipe['total_ms']:.1f}ms | Render={t_rend:.1f}ms")

    writer_main.release()
    writer_debug.release()

    actual_inf_fps = round(float(1000.0 / np.mean(latencies_total)), 2)
    valid_measurements_count = count_good + count_warning

    mean_depth = round(float(np.mean(all_measured_depths)), 1) if all_measured_depths else 0.0
    max_depth = round(float(np.max(all_measured_depths)), 1) if all_measured_depths else 0.0
    mean_area = round(float(np.mean(all_measured_areas)), 3) if all_measured_areas else 0.0
    max_area = round(float(np.max(all_measured_areas)), 3) if all_measured_areas else 0.0

    perf = {
        "sequence_name": "pothole_long_stereo_reel",
        "dataset_source": "PavementDamagesG-7 (ZED 2i Stereo Dashcam Reel)",
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
        "depth_metrics_cm": {
            "mean_depth_cm": mean_depth,
            "max_depth_cm": max_depth
        },
        "area_metrics_m2": {
            "mean_area_m2": mean_area,
            "max_area_m2": max_area
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

    perf_file = DEMO_DIR / "pothole_long_performance.json"
    with open(perf_file, "w", encoding="utf-8") as pf:
        json.dump(perf, pf, indent=4)

    print("\n------------------------------------------------------------")
    print("LONG POTHOLE STEREO VIDEO GENERATION COMPLETE:")
    print("------------------------------------------------------------")
    print(f"Dataset Source                : {perf['dataset_source']}")
    print(f"Actual Video Duration         : {actual_duration:.2f} seconds")
    print(f"Number of Frames Processed    : {len(left_files)}")
    print(f"Video Playback FPS            : {fps_target:.3f} FPS")
    print(f"Inference Processing FPS      : {actual_inf_fps:.2f} FPS")
    print(f"Unique Potholes Tracked       : {len(seen_track_ids)}")
    print(f"Total Pothole Detections      : {total_detections_count}")
    print(f"Valid Depth/Area Measurements : {valid_measurements_count} (GOOD: {count_good}, WARN: {count_warning})")
    print(f"Depth Summary                 : Mean={mean_depth} cm, Max={max_depth} cm")
    print(f"Area Summary                  : Mean={mean_area} m2, Max={max_area} m2")
    print(f"Main Demo Video Path          : {main_video_path} ({main_video_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Debug 2x2 Video Path          : {debug_video_path} ({debug_video_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Performance JSON              : {perf_file}")
    print("============================================================\n")

    return perf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo_backend", type=str, default="fast_foundation", help="Stereo model backend")
    parser.add_argument("--demo_name", type=str, default="pothole_long_stereo_demo.mp4", help="Main demo filename")
    parser.add_argument("--debug_name", type=str, default="pothole_long_stereo_debug.mp4", help="Debug demo filename")
    parser.add_argument("--duration", type=float, default=15.0, help="Target video duration in seconds")
    parser.add_argument("--fps", type=float, default=5.0, help="Explicit playback FPS override")
    args = parser.parse_args()

    generate_long_stereo_demo(
        stereo_backend=args.stereo_backend,
        output_demo_name=args.demo_name,
        output_debug_name=args.debug_name,
        target_duration_s=args.duration,
        fps_override=args.fps
    )
