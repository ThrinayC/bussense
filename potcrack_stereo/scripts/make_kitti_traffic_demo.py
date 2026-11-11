#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — KITTI Multi-Drive Stereo Traffic Demo (40–50 seconds)
=================================================================
Generates a 40–50 second production demo by concatenating MULTIPLE
real KITTI stereo sequences (all from the same calibration date).

Sources (combined):
  2011_09_26_drive_0005  — 100 frames  (~10s)
  2011_09_26_drive_0009  — up to 350 frames (~35s)
  [optional]             — additional drives if needed

All drives share the same 2011_09_26_calib → same camera parameters.
EVERY frame is a unique real stereo pair — NO interpolation or duplication.

Output:
  outputs/demo/bussense_kitti_stereo_traffic_demo.mp4    (40–50s, 1242×375)
  outputs/demo/bussense_kitti_stereo_traffic_debug.mp4   (2×2 quad, same duration)

Pipeline: Pure LEFT+RIGHT → SGBM → metric depth → YOLO → ByteTrack →
          ray/road-plane 3D → distances → density → annotated video
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

from scripts.run_combined_pipeline import BusSensePerceptionPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
cand_kitti_1 = PROJECT_ROOT.parent / "kitti_incident_intelligence" / "data" / "kitti"
cand_kitti_2 = PROJECT_ROOT.parent / "bussense" / "data" / "kitti"
KITTI_ROOT = cand_kitti_1 if cand_kitti_1.exists() else cand_kitti_2
CALIB_FILE = KITTI_ROOT / "2011_09_26_calib" / "calib_cam_to_cam.txt"
if not CALIB_FILE.exists():
    _cands = list(KITTI_ROOT.glob("**/calib_cam_to_cam.txt"))
    if _cands:
        CALIB_FILE = _cands[0]

DEMO_DIR   = PROJECT_ROOT / "outputs" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DEMO  = DEMO_DIR / "bussense_kitti_stereo_traffic_demo.mp4"
TARGET_DEBUG = DEMO_DIR / "bussense_kitti_stereo_traffic_debug.mp4"

# Target output parameters
TARGET_FPS       = 10        # 10 fps — matches KITTI native capture rate
TARGET_MIN_SECS  = 40
TARGET_MAX_SECS  = 50
OUT_W, OUT_H     = 1242, 375  # Native KITTI resolution


# ─────────────────────────────────────────────────────────────────────────────
# KITTI drive discovery
# ─────────────────────────────────────────────────────────────────────────────
def discover_drives(kitti_root: Path) -> List[Tuple[str, List[Path], List[Path]]]:
    """
    Scans KITTI_ROOT for all available drive sequences that have
    both image_02 and image_03 sub-folders.
    Returns list of (drive_name, sorted_left_frames, sorted_right_frames).
    """
    results = []

    # Check direct sub-dirs of kitti_root
    for candidate in sorted(kitti_root.iterdir()):
        if not candidate.is_dir():
            continue
        # Handle both flat layout (kitti_root/2011_09_26_drive_XXXX)
        # and nested (kitti_root/2011_09_26/2011_09_26_drive_XXXX)
        for nested in [candidate, candidate / "2011_09_26"]:
            if not nested.is_dir():
                continue
            for drive_dir in sorted(nested.iterdir()):
                if not drive_dir.is_dir() or "drive" not in drive_dir.name:
                    continue
                l_dir = drive_dir / "image_02" / "data"
                r_dir = drive_dir / "image_03" / "data"
                if l_dir.exists() and r_dir.exists():
                    lefts  = sorted(l_dir.glob("*.png"))
                    rights = sorted(r_dir.glob("*.png"))
                    if len(lefts) > 0 and len(lefts) == len(rights):
                        results.append((drive_dir.name, lefts, rights))

        # Flat layout directly
        l_dir = candidate / "image_02" / "data"
        r_dir = candidate / "image_03" / "data"
        if l_dir.exists() and r_dir.exists() and "drive" in candidate.name:
            lefts  = sorted(l_dir.glob("*.png"))
            rights = sorted(r_dir.glob("*.png"))
            if len(lefts) > 0 and len(lefts) == len(rights):
                already = any(r[0] == candidate.name for r in results)
                if not already:
                    results.append((candidate.name, lefts, rights))

    # Deduplicate by drive name (keep first occurrence)
    seen = set()
    deduped = []
    for name, l, r in results:
        if name not in seen:
            seen.add(name)
            deduped.append((name, l, r))

    return sorted(deduped, key=lambda x: x[0])


# ─────────────────────────────────────────────────────────────────────────────
# Calibration parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_kitti_calib(calib_path: Path) -> Dict[str, float]:
    with open(calib_path, "r") as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            data[key.strip()] = [float(x) for x in val.split()]
        except ValueError:
            pass
    p02 = np.array(data["P_rect_02"]).reshape(3, 4)
    p03 = np.array(data["P_rect_03"]).reshape(3, 4)
    fx = p02[0, 0]
    fy = p02[1, 1]
    cx = p02[0, 2]
    cy = p02[1, 2]
    baseline = abs(p03[0, 3]) / fx
    print(f"  Calibration: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, B={baseline:.4f}m")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline}


# ─────────────────────────────────────────────────────────────────────────────
# Colorize helpers
# ─────────────────────────────────────────────────────────────────────────────
def colorize_disparity(disp: np.ndarray, max_disp: float = 96.0) -> np.ndarray:
    valid = (disp > 0.5) & np.isfinite(disp)
    d_val = np.clip(disp, 0.0, max_disp) / max_disp * 255.0
    color = cv2.applyColorMap(d_val.astype(np.uint8), cv2.COLORMAP_PLASMA)
    color[~valid] = (20, 20, 20)
    return color


def colorize_depth(depth_m: np.ndarray, max_d: float = 50.0) -> np.ndarray:
    valid = (depth_m >= 1.0) & (depth_m <= max_d) & np.isfinite(depth_m)
    depth_norm = np.zeros_like(depth_m, dtype=np.uint8)
    if np.any(valid):
        d_val = (1.0 - (np.clip(depth_m, 1.0, max_d) - 1.0) / (max_d - 1.0)) * 255.0
        depth_norm = d_val.astype(np.uint8)
    color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
    color[~valid] = (15, 15, 15)
    return color


# ─────────────────────────────────────────────────────────────────────────────
# HUD rendering
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud_banner(
    img: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    traffic_scene: Dict[str, Any],
    density_telemetry: Dict[str, Any],
    drive_name: str,
):
    w = img.shape[1]
    banner_h = 68
    sub = img[0:banner_h, 0:w]
    bg = np.full_like(sub, (16, 20, 26))
    img[0:banner_h, 0:w] = cv2.addWeighted(bg, 0.88, sub, 0.12, 0)
    cv2.line(img, (0, banner_h), (w, banner_h), (0, 200, 255), 2)

    cv2.putText(img, "BUSSENSE  ▶  STEREO VEHICULAR SENSING  ▶  KITTI EGO VEHICLE (MOVING)",
                (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "STEREO: ACTIVE  |  TRACKER: ACTIVE  |  GT: ZERO",
                (w - 340, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 180), 1, cv2.LINE_AA)

    v_cnt = density_telemetry.get("visible_vehicle_count", 0)
    t_cnt = density_telemetry.get("tracked_vehicle_count", 0)
    ego = traffic_scene.get("ego_metrics", {})
    nearest_dist = ego.get("nearest_vehicle_distance_m")
    nearest_cls  = ego.get("nearest_vehicle_class", "")
    nearest_gap  = ego.get("nearest_longitudinal_gap_m")

    near_str = f"{nearest_dist:.1f}m ({nearest_cls})" if nearest_dist is not None else "NONE"
    gap_str  = f"{nearest_gap:.1f}m"                  if nearest_gap is not None  else "N/A"

    dens_val    = density_telemetry.get("traffic_density_veh_km2")
    dens_status = density_telemetry.get("density_status", "UNRELIABLE")
    if dens_status == "OK" and dens_val is not None:
        dens_str   = f"{dens_val:.0f} veh/km²"
        dens_color = (0, 255, 180)
    else:
        dens_str   = "UNRELIABLE"
        dens_color = (80, 100, 255)

    telem = (f"EGO: MOVING  |  VEH: {v_cnt} (Tracked: {t_cnt})"
             f"  |  NEAREST: {near_str}  |  GAP [HEURISTIC]: {gap_str}  |  DENSITY: ")
    cv2.putText(img, telem, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 240), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(telem, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(img, dens_str, (12 + tw, 50), cv2.FONT_HERSHEY_DUPLEX, 0.39, dens_color, 1, cv2.LINE_AA)

    # Bottom info strip
    h_img = img.shape[0]
    bot_sub = img[h_img - 20:h_img, 0:w]
    bot_bg  = np.full_like(bot_sub, (12, 14, 18))
    img[h_img - 20:h_img, 0:w] = cv2.addWeighted(bot_bg, 0.85, bot_sub, 0.15, 0)
    time_sec = frame_idx / max(1.0, fps)
    info = (f"Frame: {frame_idx:04d}/{total_frames:04d}  |  T: {time_sec:.1f}s  |"
            f"  Drive: {drive_name}  |  Pure Stereo  —  No GT / No LiDAR at Inference")
    cv2.putText(img, info, (12, h_img - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 175, 190), 1, cv2.LINE_AA)


def draw_vehicle_card(img: np.ndarray, veh: Dict[str, Any], sx: float, sy: float):
    h_img, w_img = img.shape[:2]
    b = veh["bbox"]
    bx1 = int(max(0, min(b[0] * sx, w_img - 1)))
    by1 = int(max(0, min(b[1] * sy, h_img - 1)))
    bx2 = int(max(0, min(b[2] * sx, w_img - 1)))
    by2 = int(max(0, min(b[3] * sy, h_img - 1)))

    disp_id  = veh.get("display_id", f"Vehicle #{veh.get('track_id', 0)}")
    cls_name = veh.get("class", "vehicle").upper()
    dist_m   = veh.get("distance_m")
    status   = veh.get("status", "OK")
    method   = veh.get("position_method", "stereo")
    X_m, Y_m, Z_m = veh.get("X_m"), veh.get("Y_m"), veh.get("Z_m")

    box_color = (0, 230, 100)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), box_color, 2)

    u_bc, v_bc = (bx1 + bx2) // 2, by2
    cv2.circle(img, (u_bc, v_bc), 4, (0, 255, 255), -1)
    cv2.circle(img, (u_bc, v_bc), 6, (0, 180, 255), 1)

    card_w, card_h = 230, 84
    card_x = max(10, min(bx1, w_img - card_w - 10))
    card_y = by1 - card_h - 8
    if card_y < 75:
        card_y = by2 + 8
    if card_y + card_h > h_img - 30:
        card_y = max(75, h_img - card_h - 30)

    card_roi = img[card_y:card_y + card_h, card_x:card_x + card_w]
    if card_roi.shape[:2] == (card_h, card_w):
        bg = np.full_like(card_roi, (20, 26, 32))
        img[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(bg, 0.85, card_roi, 0.15, 0)
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), box_color, 1)

        lx = card_x + card_w // 2
        ly = card_y + card_h if card_y < by1 else card_y
        cv2.line(img, (lx, ly), (u_bc, by1 if card_y < by1 else by2), (0, 200, 100), 1, cv2.LINE_AA)

        cv2.putText(img, f"{disp_id}  [{cls_name}]",
                    (card_x + 8, card_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        if status == "OK" and dist_m is not None:
            cv2.putText(img, f"Dist : {dist_m:.1f} m",
                        (card_x + 8, card_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA)
            if X_m is not None:
                cv2.putText(img, f"3D: ({X_m:+.1f}, {Y_m:+.1f}, {Z_m:.1f})m",
                            (card_x + 8, card_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 240), 1, cv2.LINE_AA)
            m_lbl = "Ray-Plane" if method == "ray_road_plane_intersection" else "Stereo"
            cv2.putText(img, f"Ref: {m_lbl} Ground Contact",
                        (card_x + 8, card_y + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 170, 190), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "DISTANCE: UNRELIABLE",
                        (card_x + 8, card_y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.40, (80, 100, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Verification Contact Sheet
# ─────────────────────────────────────────────────────────────────────────────
def make_multi_drive_contact_sheet(drives: List[Tuple[str, List[Path], List[Path]]], out_path: Path):
    """Saves a multi-drive contact sheet showing LEFT+RIGHT pairs for each drive."""
    rows = []
    thumb_w, thumb_h = 360, 110

    for name, lefts, rights in drives:
        mid_idx = len(lefts) // 2
        lf = cv2.imread(str(lefts[mid_idx]))
        rf = cv2.imread(str(rights[mid_idx]))
        if lf is None or rf is None:
            continue
        lf_t = cv2.resize(lf, (thumb_w, thumb_h))
        rf_t = cv2.resize(rf, (thumb_w, thumb_h))

        short_name = name.replace("2011_09_26_", "")
        cv2.putText(lf_t, f"LEFT  | {short_name} fr={mid_idx:03d}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(rf_t, f"RIGHT | {short_name} fr={mid_idx:03d}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1, cv2.LINE_AA)
        row = np.hstack([lf_t, rf_t])
        rows.append(row)

    if rows:
        sheet = np.vstack(rows)
        cv2.imwrite(str(out_path), sheet)
        print(f"  ✓ Multi-drive contact sheet → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_kitti_long_demo():
    print("\n" + "=" * 70)
    print(" BUSSENSE KITTI MULTI-DRIVE STEREO DEMO (40–50 seconds)")
    print("=" * 70)

    # 1. Discover all available drives
    drives = discover_drives(KITTI_ROOT)
    if not drives:
        raise RuntimeError(f"No KITTI drives found under {KITTI_ROOT}")

    print(f"\nDiscovered {len(drives)} KITTI drives:")
    for name, lefts, rights in drives:
        secs = len(lefts) / TARGET_FPS
        print(f"  {name}: {len(lefts)} frames ({secs:.1f}s @ {TARGET_FPS} fps)")

    # Generate verification contact sheet across all drives
    cs_path = DEMO_DIR / "kitti_multi_drive_contact_sheet.png"
    make_multi_drive_contact_sheet(drives, cs_path)

    # 2. Build frame list — concatenate drives until we reach target duration
    target_frames = TARGET_MAX_SECS * TARGET_FPS  # 500 frames for 50s
    min_frames    = TARGET_MIN_SECS * TARGET_FPS   # 400 frames for 40s

    all_left:  List[Path] = []
    all_right: List[Path] = []
    all_drive: List[str]  = []  # per-frame drive name for HUD

    for drive_name, lefts, rights in drives:
        if len(all_left) >= target_frames:
            break
        remaining = target_frames - len(all_left)
        take = min(len(lefts), remaining)
        all_left  += list(lefts[:take])
        all_right += list(rights[:take])
        all_drive += [drive_name] * take
        print(f"  Adding {take} frames from {drive_name} (total: {len(all_left)})")

    n_frames   = len(all_left)
    duration_s = n_frames / TARGET_FPS

    if n_frames < min_frames:
        print(f"\n  [WARN] Only {n_frames} frames available ({duration_s:.1f}s). "
              f"Target was {TARGET_MIN_SECS}s. Download more drives to extend.")
    else:
        print(f"\n  Total frames: {n_frames}  →  {duration_s:.1f}s @ {TARGET_FPS} fps")

    # 3. Parse calibration (shared across all 2011_09_26 drives)
    calib = parse_kitti_calib(CALIB_FILE)

    # 4. Initialize pipeline — reset between drives to avoid cross-drive ID bleed
    pipeline = BusSensePerceptionPipeline(
        calib=calib,
        stereo_model="sgbm",
        conf_pothole=0.20,
        conf_vehicle=0.20,
    )

    # 5. Video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    demo_writer  = cv2.VideoWriter(str(TARGET_DEMO),  fourcc, TARGET_FPS, (OUT_W, OUT_H))
    qw, qh = OUT_W // 2, OUT_H // 2
    debug_writer = cv2.VideoWriter(str(TARGET_DEBUG), fourcc, TARGET_FPS, (OUT_W, OUT_H))

    if not demo_writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {TARGET_DEMO}")

    perf_log = []
    t_start  = time.time()
    prev_drive = None

    print(f"\n  Rendering {n_frames} frames...")
    print(f"  Output: {OUT_W}x{OUT_H} @ {TARGET_FPS} fps\n")

    for fi, (lp, rp, drive_name) in enumerate(zip(all_left, all_right, all_drive)):

        # Reset tracker when switching drives (clean ID state)
        if drive_name != prev_drive and prev_drive is not None:
            pipeline.reset()
            print(f"  ── Switched to drive: {drive_name} (tracker reset) ──")
        prev_drive = drive_name

        left_bgr  = cv2.imread(str(lp))
        right_bgr = cv2.imread(str(rp))

        if left_bgr is None or right_bgr is None:
            print(f"  [WARN] Cannot read frame {fi}: {lp.name}")
            continue

        if left_bgr.shape != right_bgr.shape:
            right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]))

        H, W = left_bgr.shape[:2]

        # ── Run pipeline ─────────────────────────────────────────────────
        out = pipeline.process_frame(left_bgr, right_bgr, frame_index=fi)
        perf_log.append(out["runtime_ms"])

        vehicles = out["traffic"].get("vehicles", [])
        density  = out["density"]
        sx = OUT_W / W
        sy = OUT_H / H

        # ── 1. PRODUCTION DEMO FRAME ──────────────────────────────────────
        demo_frame = cv2.resize(left_bgr, (OUT_W, OUT_H))

        for v in vehicles:
            draw_vehicle_card(demo_frame, v, sx, sy)

        draw_hud_banner(
            img=demo_frame,
            frame_idx=fi + 1,
            total_frames=n_frames,
            fps=float(TARGET_FPS),
            traffic_scene=out["traffic"],
            density_telemetry=density,
            drive_name=drive_name,
        )
        demo_writer.write(demo_frame)

        # ── 2. DEBUG QUAD ─────────────────────────────────────────────────
        # Q1: LEFT + detections
        q1 = cv2.resize(left_bgr, (qw, qh))
        q1_sx, q1_sy = qw / W, qh / H
        for v in vehicles:
            b = v["bbox"]
            x1, y1, x2, y2 = (int(b[0]*q1_sx), int(b[1]*q1_sy),
                               int(b[2]*q1_sx), int(b[3]*q1_sy))
            cv2.rectangle(q1, (x1, y1), (x2, y2), (0, 230, 100), 1)
            dist = v.get("distance_m")
            vid  = v.get("display_id", "")
            lbl  = f"{vid} {dist:.1f}m" if dist is not None else vid
            cv2.putText(q1, lbl, (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(q1, f"Q1: LEFT  {drive_name}", (8, 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # Q2: RIGHT + epipolar lines
        q2 = cv2.resize(right_bgr, (qw, qh))
        for y_line in range(25, qh, 30):
            cv2.line(q2, (0, y_line), (qw, y_line), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(q2, "Q2: RIGHT + EPIPOLAR GRID", (8, 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

        # Q3: Disparity
        q3 = cv2.resize(colorize_disparity(out["disparity"]), (qw, qh))
        cv2.putText(q3, "Q3: DISPARITY MAP (PLASMA)", (8, 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 200, 0), 1, cv2.LINE_AA)

        # Q4: Depth + density telemetry
        q4 = cv2.resize(colorize_depth(out["depth_m"]), (qw, qh))
        cv2.putText(q4, "Q4: METRIC DEPTH Z=fB/d (VIRIDIS)", (8, 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (100, 255, 100), 1, cv2.LINE_AA)
        dens_v = density.get("traffic_density_veh_km2")
        dens_s = density.get("density_status", "UNRELIABLE")
        v_cnt  = density.get("visible_vehicle_count", 0)
        dens_txt = (f"Veh: {v_cnt}  Density: {dens_v:.0f} veh/km²" if (dens_s == "OK" and dens_v)
                    else f"Veh: {v_cnt}  Density: UNRELIABLE")
        cv2.putText(q4, dens_txt, (8, qh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        debug_frame = np.vstack([np.hstack([q1, q2]), np.hstack([q3, q4])])
        debug_writer.write(debug_frame)

        if (fi + 1) % 20 == 0 or (fi + 1) == n_frames:
            elapsed   = time.time() - t_start
            fps_proc  = (fi + 1) / max(0.001, elapsed)
            n_veh     = len(vehicles)
            eta       = (n_frames - fi - 1) / max(0.001, fps_proc)
            print(f"  [{fi+1:04d}/{n_frames}] drive={drive_name[-4:]}  veh={n_veh}"
                  f"  proc={fps_proc:.1f}fps  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    demo_writer.release()
    debug_writer.release()

    total_s  = time.time() - t_start
    demo_sz  = TARGET_DEMO.stat().st_size  / (1024 * 1024)
    debug_sz = TARGET_DEBUG.stat().st_size / (1024 * 1024)

    if perf_log:
        avg_total  = np.mean([p["total"]  for p in perf_log])
        avg_stereo = np.mean([p["stereo"] for p in perf_log])
        avg_veh    = np.mean([p["vehicle"] for p in perf_log])
        print(f"\n  Avg runtime/frame: stereo={avg_stereo:.0f}ms  vehicle={avg_veh:.0f}ms  total={avg_total:.0f}ms")

    perf_out = DEMO_DIR / "kitti_demo_performance.json"
    with open(perf_out, "w") as f:
        json.dump({
            "source": "KITTI multi-drive (2011_09_26) — image_02 + image_03",
            "drives_used": list(dict.fromkeys(all_drive)),
            "n_frames": n_frames,
            "target_fps": TARGET_FPS,
            "duration_s": round(duration_s, 1),
            "total_gen_time_s": round(total_s, 1),
            "avg_frame_ms": round(np.mean([p["total"] for p in perf_log]), 1) if perf_log else None,
            "demo_mb":  round(demo_sz, 2),
            "debug_mb": round(debug_sz, 2),
        }, f, indent=2)

    print(f"\n  ✓ Production Demo : {TARGET_DEMO}  ({demo_sz:.2f} MB, {duration_s:.1f}s)")
    print(f"  ✓ Debug Quad      : {TARGET_DEBUG}  ({debug_sz:.2f} MB)")
    print(f"  ✓ Performance log : {perf_out}")
    print(f"  Total generation time: {total_s:.1f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_kitti_long_demo()
