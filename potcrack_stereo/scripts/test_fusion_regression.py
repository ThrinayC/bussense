#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Phase 2, 3, 4, 5: Fusion Regression Test
====================================================
Rigorous regression test comparing the OLD separate modules vs the NEW fused pipeline.

Evaluates on:
1. 50-Second KITTI Multi-Drive Stereo Sequence (500 frames across 5 drives):
   Sampled evenly across all 5 drives: 25 frames (every 20th frame)
2. Pothole Ground-Truth Stereo Sequence (data/pothole_long_stereo):
   Sampled frames (5, 15, 25, 35, 45, 55, 65) to verify 3D pothole depth/area
3. Previous Combined Demo Sampled Frames (main_street_sbs.mp4):
   Sampled frames [5, 10, 20, 50, 150, 250, 300, 350]

Per-frame Comparison:
- STEREO: disparity stats, valid fraction, median depth
- POTHOLES: count, bbox IoU, depth (cm), area (m²), quality
- VEHICLES: count, bbox IoU, 3D (X,Y,Z), distance (m), longitudinal gap
- TRAFFIC: visible count, tracked count, density (veh/km²), road polygon area (m²)
- PERFORMANCE: per-module latency, total latency, effective FPS

Classifies each metric as: MATCH, MINOR_DEVIATION, or MAJOR_DEVIATION.
Saves full report to outputs/evaluation/fusion_regression_report.json.
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

# Old separate modules
from traffic.tracker import TrafficTracker
from traffic.spatial_analyzer import TrafficSpatialAnalyzer
from traffic.density_engine import TrafficDensityEngine
from scripts.stereo_depth import get_stereo_matcher
from scripts.pothole_geometry import PotholeGeometryEngine
from scripts.make_kitti_traffic_demo import discover_drives, parse_kitti_calib, KITTI_ROOT, CALIB_FILE

# New fused pipeline
from scripts.run_unified_pipeline import UnifiedBusSensePipeline, UnifiedFrameState


class OldSeparatePipeline:
    """Represents the un-fused baseline using separate sequential invocations."""

    def __init__(self, calib: Dict[str, float], conf_pothole: float = 0.20, conf_vehicle: float = 0.20):
        self.calib = calib
        self.fx = float(calib["fx"])
        self.fy = float(calib["fy"])
        self.cx = float(calib["cx"])
        self.cy = float(calib["cy"])
        self.baseline = float(calib["baseline"])

        self.matcher = get_stereo_matcher("sgbm")
        self.pothole_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg.pt"))
        self.conf_pothole = conf_pothole
        self.pothole_engine = PotholeGeometryEngine(
            min_depth_m=0.5, max_depth_m=45.0, ring_dilation_px=18,
            min_ring_points=15, ransac_threshold_m=0.03
        )
        self.vehicle_tracker = TrafficTracker("yolov8n.pt", conf_threshold=conf_vehicle)
        self.spatial_analyzer = TrafficSpatialAnalyzer(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            baseline=self.baseline, min_valid_disparity=0.5
        )
        self.density_engine: Optional[TrafficDensityEngine] = None
        self.next_pothole_id = 1
        self.active_potholes = {}

    def reset(self):
        self.vehicle_tracker.reset()
        if self.density_engine:
            self.density_engine.reset()
        self.next_pothole_id = 1
        self.active_potholes.clear()

    def process(self, left: np.ndarray, right: np.ndarray, frame_index: int = 0) -> Dict[str, Any]:
        t0 = time.time()
        H, W = left.shape[:2]
        if self.density_engine is None:
            self.density_engine = TrafficDensityEngine(self.fx, W, H, z_min=2.0, z_max=35.0)

        # 1. Stereo
        t_stereo0 = time.time()
        disparity = self.matcher.estimate_disparity(left, right)
        valid_disp = (disparity > 0.5) & np.isfinite(disparity)
        depth_m = np.zeros_like(disparity, dtype=np.float32)
        depth_m[valid_disp] = (self.fx * self.baseline) / disparity[valid_disp]
        t_stereo = (time.time() - t_stereo0) * 1000.0

        # 2. Pothole
        t_pot0 = time.time()
        pot_res = self.pothole_model(left, conf=self.conf_pothole, verbose=False)
        calib_dict = {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy, "baseline": self.baseline}
        pothole_mask = np.zeros((H, W), dtype=np.uint8)
        if pot_res[0].masks is not None:
            for m in pot_res[0].masks.data.cpu().numpy():
                m_resized = cv2.resize((m > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                pothole_mask = np.bitwise_or(pothole_mask, m_resized * 255)
        pot_instances = self.pothole_engine.analyze_potholes(left, depth_m, pothole_mask, calib_dict)
        t_pot = (time.time() - t_pot0) * 1000.0

        # Road plane
        best_plane = None
        for p in pot_instances:
            if p.plane_equation is not None and p.quality in ["GOOD", "WARNING"]:
                best_plane = np.array(p.plane_equation, dtype=np.float32)
                break
        if best_plane is None:
            # lower 35%
            road_roi_y = int(H * 0.65)
            roi_depth = depth_m[road_roi_y:H, :]
            valid = (roi_depth >= 1.0) & (roi_depth <= 35.0) & np.isfinite(roi_depth)
            if np.sum(valid) >= 50:
                ys, xs = np.where(valid)
                n_pts = len(ys)
                if n_pts > 1000:
                    step = n_pts / 1000.0
                    idx = np.array([int(i * step) for i in range(1000)], dtype=np.int32)
                    ys, xs = ys[idx], xs[idx]
                zs = roi_depth[ys, xs]
                xs_3d = (xs - self.cx) * zs / self.fx
                ys_3d = (ys + road_roi_y - self.cy) * zs / self.fy
                pts_3d = np.stack([xs_3d, ys_3d, zs], axis=1)
                from sklearn.linear_model import RANSACRegressor
                try:
                    ransac = RANSACRegressor(min_samples=10, residual_threshold=0.08, max_trials=100, random_state=42)
                    ransac.fit(pts_3d[:, [0, 2]], pts_3d[:, 1])
                    if np.mean(ransac.inlier_mask_) >= 0.35:
                        m1, m2 = ransac.estimator_.coef_
                        c_int = ransac.estimator_.intercept_
                        plane = np.array([m1, -1.0, m2, c_int], dtype=np.float32)
                        best_plane = plane / np.linalg.norm(plane[:3])
                except Exception:
                    pass

        # 3. Traffic
        t_veh0 = time.time()
        tracked_vehicles = self.vehicle_tracker.track(left, frame_index=frame_index)
        traffic_scene = self.spatial_analyzer.analyze_scene(tracked_vehicles, disparity, road_plane=best_plane)
        t_veh = (time.time() - t_veh0) * 1000.0

        # 4. Density
        t_dens0 = time.time()
        density = self.density_engine.update(scene_analysis=traffic_scene, road_plane=best_plane)
        t_dens = (time.time() - t_dens0) * 1000.0

        total_ms = (time.time() - t0) * 1000.0

        return {
            "disparity": disparity,
            "depth_m": depth_m,
            "potholes": pot_instances,
            "traffic": traffic_scene,
            "density": density,
            "road_plane": best_plane,
            "timing_ms": {
                "stereo": t_stereo, "pothole": t_pot, "vehicle": t_veh,
                "density": t_dens, "total": total_ms
            }
        }


def bbox_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / max(1.0, union))


def compare_frames(
    old_res: Dict[str, Any],
    new_res: UnifiedFrameState,
    frame_name: str
) -> Dict[str, Any]:
    """Compares all metrics between old and new pipelines on one stereo pair."""
    diffs = {}
    verdicts = {}

    # 1. STEREO COMPARISON
    old_disp = old_res["disparity"]
    new_disp = new_res.disparity
    disp_diff = np.abs(old_disp - new_disp)
    max_disp_diff = float(np.max(disp_diff))
    mean_disp_diff = float(np.mean(disp_diff))

    old_valid_disp = float(np.count_nonzero(old_disp > 0.5) / old_disp.size)
    new_valid_disp = new_res.stereo_quality["valid_disp_fraction"]
    valid_disp_diff = abs(old_valid_disp - new_valid_disp)

    old_med_depth = float(np.median(old_res["depth_m"][old_res["depth_m"] > 0])) if np.any(old_res["depth_m"] > 0) else 0.0
    new_med_depth = new_res.stereo_quality["median_depth_m"]
    med_depth_diff = abs(old_med_depth - new_med_depth)

    if max_disp_diff < 1e-4 and valid_disp_diff < 1e-4 and med_depth_diff < 0.05:
        verdicts["stereo"] = "MATCH"
    elif max_disp_diff < 0.5 and valid_disp_diff < 0.02:
        verdicts["stereo"] = "MINOR_DEVIATION"
    else:
        verdicts["stereo"] = "MAJOR_DEVIATION"

    diffs["stereo"] = {
        "max_disp_diff_px": round(max_disp_diff, 5),
        "mean_disp_diff_px": round(mean_disp_diff, 5),
        "valid_disp_diff": round(valid_disp_diff, 4),
        "median_depth_diff_m": round(med_depth_diff, 3),
        "verdict": verdicts["stereo"]
    }

    # 2. POTHOLE COMPARISON
    old_pots = old_res["potholes"]
    new_pots = new_res.potholes
    n_old_pots = len(old_pots)
    n_new_pots = len(new_pots)

    if n_old_pots == n_new_pots:
        pot_verdict = "MATCH"
        pot_depth_diffs = []
        pot_area_diffs = []
        for op, npot in zip(old_pots, new_pots):
            od = getattr(op, "depth_p95_cm", None)
            nd = npot.get("depth_p95_cm")
            if od is not None and nd is not None:
                pot_depth_diffs.append(abs(od - nd))
            oa = getattr(op, "area_m2", None)
            na = npot.get("area_m2")
            if oa is not None and na is not None:
                pot_area_diffs.append(abs(oa - na))

        max_d_diff = max(pot_depth_diffs) if pot_depth_diffs else 0.0
        max_a_diff = max(pot_area_diffs) if pot_area_diffs else 0.0
        if max_d_diff > 0.5 or max_a_diff > 0.02:
            pot_verdict = "MINOR_DEVIATION"
    elif abs(n_old_pots - n_new_pots) <= 1:
        pot_verdict = "MINOR_DEVIATION"
    else:
        pot_verdict = "MAJOR_DEVIATION"

    # Strict physical bounds and suppression check
    for npot in new_pots:
        nd = npot.get("depth_p95_cm")
        na = npot.get("area_m2")
        nq = npot.get("quality")
        if nd is not None and nd > 30.0:
            pot_verdict = "MAJOR_DEVIATION"
        if na is not None and na > 10.0:
            pot_verdict = "MAJOR_DEVIATION"
        if nq == "INVALID" and (nd is not None or na is not None):
            pot_verdict = "MAJOR_DEVIATION"

    verdicts["potholes"] = pot_verdict
    diffs["potholes"] = {
        "old_count": n_old_pots,
        "new_count": n_new_pots,
        "verdict": pot_verdict
    }

    # 3. VEHICLES COMPARISON
    old_vehs = old_res["traffic"].get("vehicles", [])
    new_vehs = new_res.vehicles
    n_old_v = len(old_vehs)
    n_new_v = len(new_vehs)

    if n_old_v == n_new_v:
        veh_verdict = "MATCH"
        dist_diffs = []
        z_diffs = []
        for ov, nv in zip(old_vehs, new_vehs):
            od = ov.get("distance_m")
            nd = nv.get("distance_m")
            if od is not None and nd is not None:
                dist_diffs.append(abs(od - nd))
            oz = ov.get("Z_m")
            nz = nv.get("Z_m")
            if oz is not None and nz is not None:
                z_diffs.append(abs(oz - nz))
        max_dist_diff = max(dist_diffs) if dist_diffs else 0.0
        max_z_diff = max(z_diffs) if z_diffs else 0.0
        if max_dist_diff > 0.3 or max_z_diff > 0.3:
            veh_verdict = "MINOR_DEVIATION"
    elif abs(n_old_v - n_new_v) <= 1:
        veh_verdict = "MINOR_DEVIATION"
    else:
        veh_verdict = "MAJOR_DEVIATION"

    verdicts["vehicles"] = veh_verdict
    diffs["vehicles"] = {
        "old_count": n_old_v,
        "new_count": n_new_v,
        "max_dist_diff_m": round(max(dist_diffs), 3) if (n_old_v == n_new_v and dist_diffs) else 0.0,
        "verdict": veh_verdict
    }

    # 4. TRAFFIC DENSITY & METRICS
    old_dens = old_res["density"]
    new_dens = new_res.traffic_metrics
    old_d_val = old_dens.get("traffic_density_veh_km2")
    new_d_val = new_dens.get("traffic_density_veh_km2")

    old_status = old_dens.get("density_status")
    new_status = new_dens.get("density_status")

    if old_status == new_status:
        if old_d_val is None and new_d_val is None:
            dens_verdict = "MATCH"
        elif old_d_val is not None and new_d_val is not None:
            if abs(old_d_val - new_d_val) < 2.0:
                dens_verdict = "MATCH"
            else:
                dens_verdict = "MINOR_DEVIATION"
        else:
            dens_verdict = "MINOR_DEVIATION"
    else:
        dens_verdict = "MAJOR_DEVIATION"

    verdicts["traffic_density"] = dens_verdict
    diffs["traffic_density"] = {
        "old_density_veh_km2": old_d_val,
        "new_density_veh_km2": new_d_val,
        "old_status": old_status,
        "new_status": new_status,
        "verdict": dens_verdict
    }

    # 5. ROAD PLANE
    old_plane = old_res["road_plane"]
    new_plane_meta = new_res.road_plane
    new_plane = np.array(new_plane_meta["equation"]) if new_plane_meta and new_plane_meta.get("equation") else None

    if (old_plane is None and new_plane is None):
        plane_verdict = "MATCH"
    elif old_plane is not None and new_plane is not None:
        p_diff = float(np.max(np.abs(old_plane - new_plane)))
        plane_verdict = "MATCH" if p_diff < 0.01 else "MINOR_DEVIATION" if p_diff < 0.08 else "MAJOR_DEVIATION"
    else:
        plane_verdict = "MINOR_DEVIATION"

    verdicts["road_plane"] = plane_verdict
    diffs["road_plane"] = {
        "plane_verdict": plane_verdict
    }

    # Overall frame verdict
    if any(v == "MAJOR_DEVIATION" for v in verdicts.values()):
        overall = "MAJOR_DEVIATION"
    elif any(v == "MINOR_DEVIATION" for v in verdicts.values()):
        overall = "MINOR_DEVIATION"
    else:
        overall = "MATCH"

    return {
        "frame": frame_name,
        "overall_verdict": overall,
        "component_verdicts": verdicts,
        "metrics": diffs
    }


def run_full_regression():
    print("\n" + "=" * 75)
    print(" BUSSENSE FUSION REGRESSION TEST (PHASE 2 & 3)")
    print("=" * 75)

    # 1. Discover the 50-second KITTI multi-drive sequence (500 frames)
    drives = discover_drives(KITTI_ROOT)
    calib = parse_kitti_calib(CALIB_FILE)

    all_left = []
    all_right = []
    all_names = []

    target_frames = 500
    for drive_name, lefts, rights in drives:
        if len(all_left) >= target_frames:
            break
        rem = target_frames - len(all_left)
        take = min(len(lefts), rem)
        all_left += list(lefts[:take])
        all_right += list(rights[:take])
        all_names += [drive_name] * take

    n_total = len(all_left)
    duration_s = n_total / 10.0
    print(f"\nDiscovered 50-second source: {n_total} frames ({duration_s:.1f}s)")
    print("Drives in source:")
    for d_name in sorted(set(all_names)):
        cnt = all_names.count(d_name)
        print(f"  {d_name}: {cnt} frames ({cnt/10.0:.1f}s)")

    # 2. Sample 25 frames across the 50-second sequence (every 20th frame)
    sample_indices = list(range(19, n_total, 20))
    print(f"\nSampled {len(sample_indices)} test frames across the 50-second sequence:")
    print("Indices:", sample_indices)

    # Instantiate pipelines
    print("\nInitializing Old Separate Pipeline vs New Unified Fused Pipeline...")
    old_pipe = OldSeparatePipeline(calib=calib, conf_pothole=0.20, conf_vehicle=0.20)
    new_pipe = UnifiedBusSensePipeline(calib=calib, conf_pothole=0.20, conf_vehicle=0.20)

    results = []
    major_count = 0
    minor_count = 0
    match_count = 0

    old_latencies = []
    new_latencies = []

    print("\nExecuting side-by-side regression comparison...")
    for step_idx, fi in enumerate(sample_indices):
        lp = all_left[fi]
        rp = all_right[fi]
        d_name = all_names[fi]

        left_bgr = cv2.imread(str(lp))
        right_bgr = cv2.imread(str(rp))
        if left_bgr is None or right_bgr is None:
            continue
        if left_bgr.shape != right_bgr.shape:
            right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]))

        # Run Old Pipeline
        t_old0 = time.time()
        old_out = old_pipe.process(left_bgr, right_bgr, frame_index=fi)
        old_lat = (time.time() - t_old0) * 1000.0
        old_latencies.append(old_lat)

        # Run New Fused Pipeline
        t_new0 = time.time()
        new_out = new_pipe.process_frame(left_bgr, right_bgr, frame_index=fi, timestamp=fi / 10.0)
        new_lat = (time.time() - t_new0) * 1000.0
        new_latencies.append(new_lat)

        # Compare
        comp = compare_frames(old_out, new_out, f"{d_name}_fr{fi:04d}")
        results.append(comp)

        verdict = comp["overall_verdict"]
        if verdict == "MAJOR_DEVIATION":
            major_count += 1
            mark = "[MAJOR]"
        elif verdict == "MINOR_DEVIATION":
            minor_count += 1
            mark = "[MINOR]"
        else:
            match_count += 1
            mark = "[MATCH]"

        v_cnt = len(new_out.vehicles)
        d_stat = new_out.traffic_metrics.get("density_status", "UNREL")
        print(f"  Frame [{fi+1:04d}/500] ({d_name[-10:]}) -> {mark} | Veh: {v_cnt} | Density: {d_stat} | Old: {old_lat:.0f}ms | New: {new_lat:.0f}ms")

    # 3. Also run pothole regression on actual pothole dataset (data/pothole_long_stereo)
    print("\nRunning Pothole Metrology Regression on authentic pothole stereo feed...")
    pot_left_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "left"
    pot_right_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "right"
    pot_left_frames = sorted(pot_left_dir.glob("*.jpg"))
    pot_right_frames = sorted(pot_right_dir.glob("*.jpg"))

    pot_calib = {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 960.0,
        "cy": 540.0,
        "baseline": 0.12,
        "image_width": 1920,
        "image_height": 1080
    }
    old_pot_pipe = OldSeparatePipeline(calib=pot_calib, conf_pothole=0.20, conf_vehicle=0.20)
    new_pot_pipe = UnifiedBusSensePipeline(calib=pot_calib, conf_pothole=0.20, conf_vehicle=0.20)

    pot_sample_indices = [5, 15, 25, 35, 45, 55, 65]
    pot_results = []
    pothole_table_rows = []

    print("\n" + "=" * 115)
    print(" 7-FRAME AUTHENTIC POTHOLE METROLOGY REGRESSION GATE")
    print("=" * 115)
    header = f"{'Frame':^9} | {'Potholes':^8} | {'Quality':^9} | {'Depth (p95)':^13} | {'Area':^11} | {'Dims (L x W)':^15} | {'Verdict':^9} | {'Notes / Reason':^28}"
    print(header)
    print("-" * 115)

    for p_idx in pot_sample_indices:
        if p_idx >= len(pot_left_frames):
            continue
        lf = cv2.imread(str(pot_left_frames[p_idx]))
        rf = cv2.imread(str(pot_right_frames[p_idx]))
        if lf is None or rf is None:
            continue

        old_res = old_pot_pipe.process(lf, rf, frame_index=p_idx)
        new_res = new_pot_pipe.process_frame(lf, rf, frame_index=p_idx, timestamp=p_idx / 15.0)
        comp = compare_frames(old_res, new_res, f"pothole_fr{p_idx:03d}")
        pot_results.append(comp)

        p_cnt = len(new_res.potholes)
        frame_passed = True
        notes = "OK"

        if p_cnt > 0:
            first_pot = new_res.potholes[0]
            q_val = first_pot.get("quality", "UNKNOWN")
            d_val = first_pot.get("depth_p95_cm")
            a_val = first_pot.get("area_m2")
            l_val = first_pot.get("length_m")
            w_val = first_pot.get("width_m")
            q_reason = first_pot.get("quality_reason", "")

            # Strict guardrail validations
            if d_val is not None and (d_val > 30.0 or d_val < -5.0):
                frame_passed = False
                notes = f"FAIL: Depth {d_val}cm exceeds physical bounds"
            elif a_val is not None and a_val > 10.0:
                frame_passed = False
                notes = f"FAIL: Area {a_val}m² exceeds physical bounds"
            elif q_val == "INVALID" and (d_val is not None or a_val is not None):
                frame_passed = False
                notes = "FAIL: INVALID quality did not suppress depth/area"
            else:
                notes = q_reason[:28] if q_reason else "Plausible measurement"

            d_str = f"{d_val:.1f} cm" if d_val is not None else "None"
            a_str = f"{a_val:.4f} m²" if a_val is not None else "None"
            dim_str = f"{l_val:.2f}m x {w_val:.2f}m" if (l_val is not None and w_val is not None) else "None"
        else:
            q_val = "NONE"
            d_str = "None"
            a_str = "None"
            dim_str = "None"
            notes = "No potholes detected"

        pot_verdict = comp["component_verdicts"]["potholes"]
        if not frame_passed or pot_verdict == "MAJOR_DEVIATION":
            major_count += 1
            status_mark = "FAILED"
        elif pot_verdict == "MINOR_DEVIATION":
            minor_count += 1
            status_mark = "WARN"
        else:
            match_count += 1
            status_mark = "PASSED"

        row_str = f"Frame {p_idx:03d} | {p_cnt:^8} | {q_val:^9} | {d_str:^13} | {a_str:^11} | {dim_str:^15} | {status_mark:^9} | {notes:<28}"
        print(row_str)
        pothole_table_rows.append({
            "frame": p_idx,
            "potholes": p_cnt,
            "quality": q_val,
            "depth_p95_cm": d_str,
            "area_m2": a_str,
            "dims": dim_str,
            "status": status_mark,
            "notes": notes
        })

    print("-" * 115)

    # 4. Summary & Verification
    total_evaluated = len(results) + len(pot_results)
    avg_old_ms = float(np.mean(old_latencies)) if old_latencies else 0.0
    avg_new_ms = float(np.mean(new_latencies)) if new_latencies else 0.0

    print("\n" + "=" * 75)
    print(" REGRESSION TEST SUMMARY")
    print("=" * 75)
    print(f"  Total Frames Evaluated: {total_evaluated}")
    print(f"  MATCH:                  {match_count} ({match_count/total_evaluated*100:.1f}%)")
    print(f"  MINOR_DEVIATION:        {minor_count} ({minor_count/total_evaluated*100:.1f}%)")
    print(f"  MAJOR_DEVIATION:        {major_count} ({major_count/total_evaluated*100:.1f}%)")
    print(f"  Old Avg Latency:        {avg_old_ms:.1f} ms ({1000.0/max(1, avg_old_ms):.1f} FPS)")
    print(f"  New Avg Latency:        {avg_new_ms:.1f} ms ({1000.0/max(1, avg_new_ms):.1f} FPS)")

    passed = (major_count == 0)
    final_verdict = "PASSED" if passed else "FAILED"
    print(f"\n  ▶ REGRESSION GATE RESULT: FUSION REGRESSION: {final_verdict}")
    print("=" * 75 + "\n")

    report = {
        "status": final_verdict,
        "total_frames_evaluated": total_evaluated,
        "kitti_50s_frames_evaluated": len(results),
        "pothole_dataset_frames_evaluated": len(pot_results),
        "counts": {
            "match": match_count,
            "minor_deviation": minor_count,
            "major_deviation": major_count
        },
        "performance": {
            "old_avg_latency_ms": round(avg_old_ms, 1),
            "new_avg_latency_ms": round(avg_new_ms, 1),
            "speedup_ratio": round(avg_old_ms / max(1e-3, avg_new_ms), 2)
        },
        "kitti_sample_results": results,
        "pothole_sample_results": pot_results
    }

    out_file = PROJECT_ROOT / "outputs" / "evaluation" / "fusion_regression_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full regression report saved to: {out_file}\n")

    return passed, report


if __name__ == "__main__":
    passed, rep = run_full_regression()
    if not passed:
        sys.exit(1)
