#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Mega Master Video Demonstration Renderer
====================================================
Generates the comprehensive 3–5 minute continuous demonstration:
1. outputs/demo/BUSSENSE_MEGA_FINAL.mp4
   - Continuous full-screen perception stream across all verified scenarios
   - Zero title cards, zero intro cards, zero section cards, zero blank transitions
   - Persistent top telemetry header, bottom road status bar, and lower-corner source badge
   - Strict vehicle bbox HUD rule: plate text shown ONLY when OCR verified; NO unreadable text
2. outputs/demo/BUSSENSE_MEGA_FINAL_DEBUG.mp4
   - 4-Pane layout (1280x720):
     * Top-Left: RGB Scene + HUD (640x360)
     * Top-Right: Real Stereo Disparity Heatmap (640x360)
     * Bottom-Left: BEV Trajectory & Corridor Radar (640x360)
     * Bottom-Right: Telemetry, Traffic Flow & Jam Risk Dashboard (640x360)
3. outputs/demo/BUSSENSE_STEREO_ONLY_FINAL.mp4
   - Strictly real stereo perception (KITTI, EvTTC, Extreme Driving, Road Surface)
4. outputs/demo/BUSSENSE_HIT_AND_RUN_MONOCULAR_FINAL.mp4
   - Strictly monocular hit-and-run evaluation
"""

import os
import sys
import math
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_hud_header(
    frame: np.ndarray,
    speed_kmh: float,
    flow_vpm: float,
    density_vkm: float,
    traffic_state: str,
    lead_gap_m: Optional[float],
    ttc_s: Optional[float],
) -> np.ndarray:
    """Draws persistent top HUD telemetry bar."""
    H, W = frame.shape[:2]
    # Semi-transparent top bar
    bar_h = 44
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.line(frame, (0, bar_h), (W, bar_h), (50, 50, 50), 1)

    # Brand badge
    cv2.rectangle(frame, (10, 8), (125, 36), (0, 120, 240), -1)
    cv2.putText(frame, "BUSSENSE", (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Metrics
    x_pos = 145
    items = [
        ("SPEED", f"{speed_kmh:.1f} km/h", (255, 255, 255)),
        ("FLOW", f"{flow_vpm:.0f} veh/min", (0, 220, 255)),
        ("DENSITY", f"{density_vkm:.0f} veh/km", (200, 255, 200)),
        ("TRAFFIC", traffic_state, (0, 80, 255) if traffic_state == "JAM RISK" else ((0, 180, 255) if traffic_state == "CONGESTED" else (0, 220, 0))),
        ("LEAD GAP", f"{lead_gap_m:.1f} m" if lead_gap_m is not None else "--", (255, 255, 255)),
        ("TTC", f"{ttc_s:.2f} s" if ttc_s is not None else "--", (0, 0, 255) if (ttc_s and ttc_s <= 2.0) else (0, 220, 255)),
    ]
    for lbl, val, col in items:
        cv2.putText(frame, lbl, (x_pos, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, val, (x_pos, 34), cv2.FONT_HERSHEY_DUPLEX, 0.48, col, 1, cv2.LINE_AA)
        x_pos += 155

    return frame


def draw_source_badge(
    frame: np.ndarray,
    source_name: str,
    stereo_active: bool = True,
    scenario_tag: str = "",
) -> np.ndarray:
    """Draws subtle non-intrusive lower corner badge for live source indication."""
    H, W = frame.shape[:2]
    bw, bh = 290, 52
    bx1, by1 = W - bw - 15, H - bh - 15
    bx2, by2 = W - 15, H - 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 60, 60), 1)

    s_text = f"SOURCE: {source_name.upper()}"
    m_text = "STEREO: ACTIVE (LEFT+RIGHT)" if stereo_active else "MONOCULAR TEST MODE"
    m_col = (0, 220, 0) if stereo_active else (0, 140, 255)

    cv2.putText(frame, s_text, (bx1 + 10, by1 + 18), cv2.FONT_HERSHEY_DUPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, m_text, (bx1 + 10, by1 + 34), cv2.FONT_HERSHEY_DUPLEX, 0.38, m_col, 1, cv2.LINE_AA)
    if scenario_tag:
        cv2.putText(frame, scenario_tag[:30], (bx1 + 10, by1 + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)

    return frame


def draw_vehicle_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    cls_name: str,
    range_m: Optional[float] = None,
    ttc_s: Optional[float] = None,
    plate_text: Optional[str] = None,
    is_lead: bool = False,
) -> np.ndarray:
    """
    Renders vehicle bounding box with strict HUD compliance:
    ONLY write plate registration IF successfully recognized.
    NEVER write UNREADABLE, OCR FAILED, UNKNOWN, or N/A.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    col = (0, 140, 255) if is_lead else (0, 220, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    # Corner decorations
    c_len = min(15, (x2 - x1) // 4)
    cv2.line(frame, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
    cv2.line(frame, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2)
    cv2.line(frame, (x2, y1), (x2 - c_len, y1), (255, 255, 255), 2)
    cv2.line(frame, (x2, y1), (x2, y1 + c_len), (255, 255, 255), 2)

    # Label stack above/inside
    clean_plate = None
    if plate_text and str(plate_text).strip():
        p_str = str(plate_text).strip()
        forbidden = ["UNREADABLE", "UNKNOWN", "NO PLATE", "N/A", "OCR FAILED", "NULL"]
        if not any(f in p_str.upper() for f in forbidden):
            clean_plate = p_str

    lines = [f"ID {track_id} | {cls_name.upper()}"]
    if clean_plate:
        lines.append(f"{clean_plate}")
    if range_m is not None:
        lines.append(f"{range_m:.1f}m")
    if ttc_s is not None and ttc_s < 10.0:
        lines.append(f"TTC {ttc_s:.1f}s")

    badge_h = len(lines) * 16 + 6
    badge_w = max(len(l) * 9 for l in lines) + 12
    by1 = max(0, y1 - badge_h)
    by2 = by1 + badge_h
    bx1 = x1
    bx2 = x1 + badge_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 1)

    cur_y = by1 + 14
    for idx, line in enumerate(lines):
        t_col = (0, 255, 255) if (clean_plate and line == clean_plate) else (255, 255, 255)
        cv2.putText(frame, line, (bx1 + 5, cur_y), cv2.FONT_HERSHEY_DUPLEX, 0.40, t_col, 1, cv2.LINE_AA)
        cur_y += 15

    return frame


def draw_vru_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    vru_class: str,
    range_m: Optional[float] = None,
    ttc_s: Optional[float] = None,
    in_corridor: bool = False,
) -> np.ndarray:
    """Renders VRU bounding box with cyclist/pedestrian disambiguation."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    col = (0, 0, 255) if in_corridor else (0, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    lbl = f"ID {track_id} {vru_class.upper()}"
    if range_m:
        lbl += f" | {range_m:.1f}m"
    if ttc_s and ttc_s < 5.0:
        lbl += f" | TTC {ttc_s:.1f}s"

    cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + len(lbl) * 9, y1), (20, 20, 20), -1)
    cv2.putText(frame, lbl, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    return frame


def draw_pothole_badge(
    frame: np.ndarray,
    pothole_idx: int,
    depth_cm: Optional[float],
    area_m2: Optional[float],
    center_xy: Tuple[int, int],
    is_valid: bool = True,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    has_current_support: bool = True,
) -> np.ndarray:
    """
    Draws pothole defect badge with metric depth and footprint area.
    STRICT VISUAL SUPPRESSION RULE (BusSense Requirement 6 & 7):
    If measurement is invalid or has_current_support is False:
    SUPPRESS VISUAL OVERLAY COMPLETELY.
    DO NOT RENDER BBOX. DO NOT WRITE 'DEPTH UNAVAILABLE', 'INVALID', OR 'UNKNOWN'.
    The track continues internally, but HUD displays nothing.
    """
    if not has_current_support or not is_valid or depth_cm is None or area_m2 is None or depth_cm <= 0.0 or area_m2 <= 0.0:
        return frame

    cx, cy = center_xy
    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 140, 255), 2)

    cv2.circle(frame, (cx, cy), 18, (0, 100, 255), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    lines = [
        f"POTHOLE #{pothole_idx:02d}",
        f"DEPTH {depth_cm:.2f} cm",
        f"AREA {area_m2:.3f} m2",
    ]
    bx, by = cx + 24, cy - 25
    bw, bh = 145, 48
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 140, 255), 1)

    cv2.line(frame, (cx, cy), (bx, by + bh // 2), (0, 140, 255), 1)

    y_cur = by + 13
    for l in lines:
        cv2.putText(frame, l, (bx + 6, y_cur), cv2.FONT_HERSHEY_DUPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        y_cur += 15

    return frame


def draw_bottom_status_bar(
    frame: np.ndarray,
    road_condition: str,
    pothole_count: int,
    max_depth_cm: Optional[float],
    alerts: List[Dict[str, Any]],
) -> np.ndarray:
    """Renders bottom operational status bar."""
    H, W = frame.shape[:2]
    bar_h = 32
    by1 = H - bar_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, by1), (W, H), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, by1), (W, by1), (45, 45, 45), 1)

    # Road condition badge
    rc_col = (0, 220, 0) if road_condition == "NORMAL" else ((0, 200, 255) if road_condition == "MEDIUM" else ((0, 120, 255) if road_condition == "HIGH" else (0, 0, 255)))
    cv2.putText(frame, f"ROAD CONDITION: {road_condition}", (15, by1 + 21), cv2.FONT_HERSHEY_DUPLEX, 0.45, rc_col, 1, cv2.LINE_AA)

    # Pothole summary
    p_str = f"POTHOLES: {pothole_count}"
    if max_depth_cm:
        p_str += f" (MAX DEPTH {max_depth_cm:.1f} cm)"
    cv2.putText(frame, p_str, (320, by1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Alert ticker
    if alerts:
        al = alerts[0]
        al_text = f"ALERT: {al.get('text', al.get('type', ''))}"
        al_col = al.get("color", (0, 0, 255))
        cv2.putText(frame, al_text[:50], (680, by1 + 21), cv2.FONT_HERSHEY_DUPLEX, 0.45, al_col, 1, cv2.LINE_AA)

    return frame


def render_bev_map(
    tracks: List[Any],
    lead_trk: Optional[Any],
    corridor_half_w: float = 1.8,
    max_range_m: float = 35.0,
    size: Tuple[int, int] = (640, 360),
) -> np.ndarray:
    """Renders 2D Bird's Eye View map of bus corridor and obstacle trajectories."""
    bev = np.full((size[1], size[0], 3), 18, dtype=np.uint8)
    cx = size[0] // 2
    by = size[1] - 30

    scale_y = (size[1] - 50) / max_range_m
    scale_x = scale_y

    # Distance grid
    for d in [5, 10, 15, 20, 25, 30, 35]:
        ry = int(by - d * scale_y)
        cv2.line(bev, (cx - 160, ry), (cx + 160, ry), (35, 35, 35), 1)
        cv2.putText(bev, f"{d}m", (cx + 165, ry + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1)

    # Safety corridor
    c_xl = int(cx - corridor_half_w * scale_x)
    c_xr = int(cx + corridor_half_w * scale_x)
    cv2.rectangle(bev, (c_xl, int(by - 25.0 * scale_y)), (c_xr, by), (0, 35, 0), -1)
    cv2.line(bev, (c_xl, by), (c_xl, int(by - 25.0 * scale_y)), (0, 160, 0), 1)
    cv2.line(bev, (c_xr, by), (c_xr, int(by - 25.0 * scale_y)), (0, 160, 0), 1)

    # Ego vehicle footprint
    cv2.rectangle(bev, (cx - 14, by - 28), (cx + 14, by), (0, 140, 255), -1)
    cv2.putText(bev, "EGO BUS", (cx - 24, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Obstacles
    for trk in tracks:
        pos = getattr(trk, "current_position_3d", None)
        if pos is not None:
            x, z = pos[0], pos[1]
            if 0.5 <= z <= max_range_m:
                px = int(cx + x * scale_x)
                py = int(by - z * scale_y)
                is_lead = (trk == lead_trk)
                col = (0, 140, 255) if is_lead else ((0, 255, 255) if getattr(trk, "category", "") == "vru" else (0, 220, 0))
                cv2.circle(bev, (px, py), 6, col, -1)
                cv2.putText(bev, f"#{trk.track_id}", (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

    cv2.putText(bev, "BIRD'S EYE VIEW (BEV) 3D RADAR MAP", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
    return bev


def render_telemetry_pane(
    rows: List[Tuple[str, str, Tuple[int, int, int]]],
    status_title: str,
    status_color: Tuple[int, int, int],
    size: Tuple[int, int] = (640, 360),
) -> np.ndarray:
    """Renders lower-right telemetry dashboard card."""
    pane = np.full((size[1], size[0], 3), 15, dtype=np.uint8)
    # Header card
    cv2.rectangle(pane, (15, 15), (size[0] - 15, 55), (28, 28, 28), -1)
    cv2.rectangle(pane, (15, 15), (size[0] - 15, 55), status_color, 2)
    cv2.putText(pane, status_title[:45], (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.52, status_color, 1, cv2.LINE_AA)

    y_pos = 85
    for lbl, val, col in rows:
        cv2.putText(pane, lbl, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(pane, str(val), (360, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.line(pane, (25, y_pos + 6), (size[0] - 25, y_pos + 6), (30, 30, 30), 1)
        y_pos += 30

    return pane
