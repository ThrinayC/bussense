#!/usr/bin/env python3
"""
scripts/audit_stereo_disparity_depth.py
Comprehensive audit for Parts C, F, G, H, I:
- Empirical disparity noise floor on flat road
- Inside vs road ring disparity and depth
- RANSAC local road plane fitting and signed perpendicular depth
- Ray-plane Shoelace footprint area vs bbox area
- Reprojected 3D polygons and disparity visualizations
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stereo_depth import get_stereo_matcher
from scripts.pothole_geometry import clean_pothole_candidates, PotholeGeometryEngine

def run_stereo_disparity_audit():
    print("============================================================")
    print("PARTS C, F, G, H, I — STEREO DISPARITY, ROAD PLANE & DEPTH AUDIT")
    print("============================================================")

    eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
    left_dir = data_dir / "left"
    right_dir = data_dir / "right"

    # Calibration: StereoLabs ZED 2i calibrated intrinsics for 1080p
    calib = {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 960.0,
        "cy": 540.0,
        "baseline": 0.12
    }
    fx, fy, cx, cy, B = calib["fx"], calib["fy"], calib["cx"], calib["cy"], calib["baseline"]

    # 1. Initialize SGBM Stereo Matcher (same matcher used in pipeline)
    matcher = get_stereo_matcher("sgbm")
    pothole_model = YOLO(str(PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg.pt"))
    geom_engine = PotholeGeometryEngine()

    # Step 1: Estimate empirical disparity noise floor from flat road regions
    # We inspect flat road region (y in [750, 950], x in [700, 1200]) across 5 frames with no potholes (e.g. frames 20, 25, 30, 40, 50)
    flat_frames = [20, 25, 30, 40, 50]
    flat_disps = []
    flat_residuals = []

    for f_idx in flat_frames:
        l_img = cv2.imread(str(left_dir / f"frame_{f_idx:05d}.jpg"))
        r_img = cv2.imread(str(right_dir / f"frame_{f_idx:05d}.jpg"))
        disp = matcher.estimate_disparity(l_img, r_img)
        
        # Flat road patch
        patch = disp[750:950, 700:1200]
        val = patch[patch > 0.5]
        if len(val) > 100:
            flat_disps.extend(val.tolist())
            # Fit planar trend to patch using RANSAC to compute true disparity noise floor
            vy, vx = np.where(patch > 0.5)
            vz = patch[vy, vx]
            if len(vz) > 100:
                X_fit = np.column_stack([vx, vy])
                ransac = RANSACRegressor(residual_threshold=2.0, random_state=42)
                ransac.fit(X_fit, vz)
                inliers = ransac.inlier_mask_
                if np.any(inliers):
                    res = vz[inliers] - ransac.predict(X_fit[inliers])
                    flat_residuals.extend(res.tolist())

    flat_disps = np.array(flat_disps)
    flat_residuals = np.array(flat_residuals)
    disp_noise_std = float(np.std(flat_residuals))
    disp_noise_iqr = float(np.percentile(flat_residuals, 75) - np.percentile(flat_residuals, 25))

    print(f"\n--- EMPIRICAL DISPARITY NOISE FLOOR (FLAT ROAD REGIONS) ---")
    print(f"Sampled Flat Road Pixels : {len(flat_residuals)}")
    print(f"Disparity Noise Std (sigma_d): {disp_noise_std:.3f} px")
    print(f"Disparity Noise IQR          : {disp_noise_iqr:.3f} px")
    print(f"Statistical indistinguishability threshold: 1.96 * sigma_d = {1.96 * disp_noise_std:.3f} px")

    # Step 2: Audit 5 representative pothole instances
    # Chosen frames with clear, high-confidence pothole detections:
    target_frames = [5, 7, 10, 14, 15]
    audit_results = []
    pothole_geom_diag_records = []

    # Visualizations for artifacts
    disp_vis_tiles = []
    reproject_vis_tiles = []

    for f_idx in target_frames:
        l_path = left_dir / f"frame_{f_idx:05d}.jpg"
        r_path = right_dir / f"frame_{f_idx:05d}.jpg"
        left_rgb = cv2.imread(str(l_path))
        right_rgb = cv2.imread(str(r_path))
        H, W = left_rgb.shape[:2]

        # Single stereo pass
        disp = matcher.estimate_disparity(left_rgb, right_rgb)
        valid_disp = (disp > 0.5) & np.isfinite(disp)
        depth_m = np.zeros_like(disp, dtype=np.float32)
        depth_m[valid_disp] = (fx * B) / disp[valid_disp]

        # Pothole detection
        res = pothole_model(left_rgb, conf=0.20, verbose=False)[0]
        raw_cands = []
        if res.boxes is not None and res.masks is not None:
            for i in range(len(res.boxes)):
                box = [int(v) for v in res.boxes.xyxy[i].cpu().numpy()]
                m = res.masks.data[i].cpu().numpy()
                m_res = cv2.resize((m > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                raw_cands.append({
                    "id": i + 1,
                    "bbox": tuple(box),
                    "confidence": float(res.boxes.conf[i]),
                    "mask": m_res * 255
                })

        cleaned_cands = clean_pothole_candidates(raw_cands, (H, W))
        if not cleaned_cands:
            continue

        # Choose the most prominent pothole in this frame
        chosen = max(cleaned_cands, key=lambda c: cv2.countNonZero(c["mask"]))
        pot_mask = chosen["mask"]
        cnts, _ = cv2.findContours(pot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        # 1. Road ring: 18-30px dilation excluding pothole
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        dilated = cv2.dilate(pot_mask, kernel, iterations=1)
        road_ring = cv2.bitwise_and(dilated, cv2.bitwise_not(pot_mask))

        # Disparity inside pothole
        in_v, in_u = np.where(pot_mask > 0)
        in_disp_all = disp[in_v, in_u]
        in_val = in_disp_all > 0.5
        in_disp = in_disp_all[in_val]
        in_frac = float(len(in_disp) / max(1, len(in_v)))

        # Disparity on road ring
        rd_v, rd_u = np.where(road_ring > 0)
        rd_disp_all = disp[rd_v, rd_u]
        rd_val = rd_disp_all > 0.5
        rd_disp = rd_disp_all[rd_val]
        rd_frac = float(len(rd_disp) / max(1, len(rd_v)))

        # Telemetry
        in_mean = float(np.mean(in_disp)) if len(in_disp) > 0 else 0.0
        in_med = float(np.median(in_disp)) if len(in_disp) > 0 else 0.0
        in_min = float(np.min(in_disp)) if len(in_disp) > 0 else 0.0
        in_max = float(np.max(in_disp)) if len(in_disp) > 0 else 0.0

        rd_mean = float(np.mean(rd_disp)) if len(rd_disp) > 0 else 0.0
        rd_med = float(np.median(rd_disp)) if len(rd_disp) > 0 else 0.0
        rd_min = float(np.min(rd_disp)) if len(rd_disp) > 0 else 0.0
        rd_max = float(np.max(rd_disp)) if len(rd_disp) > 0 else 0.0

        delta_d = float(rd_med - in_med)
        
        # Depths
        in_z = depth_m[in_v, in_u]
        in_z_val = in_z[(in_z >= 0.5) & (in_z <= 45.0) & np.isfinite(in_z)]
        in_z_med = float(np.median(in_z_val)) if len(in_z_val) > 0 else 0.0

        rd_z = depth_m[rd_v, rd_u]
        rd_z_val = rd_z[(rd_z >= 0.5) & (rd_z <= 45.0) & np.isfinite(rd_z)]
        rd_z_med = float(np.median(rd_z_val)) if len(rd_z_val) > 0 else 0.0

        delta_z_m = in_z_med - rd_z_med
        delta_z_cm = delta_z_m * 100.0

        # Theoretical delta_z from disparity: Delta_Z = fB/d_inside - fB/d_road
        if in_med > 0.5 and rd_med > 0.5:
            expected_delta_z_cm = ((fx * B / in_med) - (fx * B / rd_med)) * 100.0
        else:
            expected_delta_z_cm = 0.0

        # Road plane fitting on road ring
        plane_eq, inlier_ratio, rmse_m, vert_angle, plane_status = geom_engine.fit_road_plane(
            depth_m, pot_mask, fx, fy, cx, cy
        )

        signed_depth_cm = None
        p95_depth_cm = None
        max_depth_cm = None
        y_inside_mean = None
        y_road_mean = None
        depth_status = "NO_PLANE"

        # Unified valid points inside pothole
        valid_in = (in_disp_all > 0.5) & (in_z >= 0.5) & (in_z <= 45.0) & np.isfinite(in_z)
        valid_rd = (rd_disp_all > 0.5) & (rd_z >= 0.5) & (rd_z <= 45.0) & np.isfinite(rd_z)

        if plane_eq is not None and np.count_nonzero(valid_in) >= 5:
            a, b, c, d_pl = plane_eq
            in_u_val = in_u[valid_in]
            in_v_val = in_v[valid_in]
            in_z_val = in_z[valid_in]

            in_x = (in_u_val - cx) * in_z_val / fx
            in_y = (in_v_val - cy) * in_z_val / fy
            # Perpendicular depressions below road surface: -(aX + bY + cZ + d)
            depressions = -(a * in_x + b * in_y + c * in_z_val + d_pl)
            signed_depth_cm = float(np.mean(depressions)) * 100.0
            p95_depth_cm = float(np.percentile(np.maximum(depressions, 0.0), 95)) * 100.0
            max_depth_cm = float(np.max(depressions)) * 100.0
            y_inside_mean = float(np.mean(in_y))
            
            # Diagnostic telemetry: Y in road ring
            if np.count_nonzero(valid_rd) >= 5:
                rd_u_val = rd_u[valid_rd]
                rd_v_val = rd_v[valid_rd]
                rd_z_val = rd_z[valid_rd]
                rd_y = (rd_v_val - cy) * rd_z_val / fy
                y_road_mean = float(np.mean(rd_y))

            depth_status = "OK"

        # Area calculation (Ray / plane intersection + Shoelace)
        area_m2 = None
        length_m = None
        width_m = None
        bbox_area_m2 = None
        polygon_pts_3d = []
        polygon_reprojected_2d = []

        if plane_eq is not None:
            area_m2, length_m, width_m, _, area_status = geom_engine.compute_footprint_area_rays(
                cnt.reshape(-1, 2), plane_eq, fx, fy, cx, cy, z_med=in_z_med
            )
            if length_m is not None and width_m is not None:
                bbox_area_m2 = length_m * width_m

            # Reproject 3D polygon back to image coordinates for verification
            a, b, c, d_pl = plane_eq
            pts_2d = cnt.reshape(-1, 2)
            for u_px, v_px in pts_2d:
                dx = (u_px - cx) / fx
                dy = (v_px - cy) / fy
                denom = a * dx + b * dy + c
                if abs(denom) > 1e-6:
                    lambda_val = -d_pl / denom
                    if 0.5 <= lambda_val <= 45.0:
                        X = lambda_val * dx
                        Y = lambda_val * dy
                        Z = lambda_val
                        polygon_pts_3d.append([X, Y, Z])
                        # Reproject to 2D
                        u_rep = int(round(fx * X / Z + cx))
                        v_rep = int(round(fy * Y / Z + cy))
                        polygon_reprojected_2d.append([u_rep, v_rep])

        # Statistical distinguishability check (User Correction 1)
        statistically_distinguishable = abs(delta_d) > (1.96 * disp_noise_std)

        record = {
            "frame_index": f_idx,
            "filename": f"frame_{f_idx:05d}.jpg",
            "pothole_id": chosen["id"],
            "confidence": float(chosen["confidence"]),
            "mask_pixel_count": int(cv2.countNonZero(pot_mask)),
            "disparity_inside": {
                "mean": round(in_mean, 2), "median": round(in_med, 2),
                "min": round(in_min, 2), "max": round(in_max, 2),
                "valid_fraction": round(in_frac, 3)
            },
            "disparity_road": {
                "mean": round(rd_mean, 2), "median": round(rd_med, 2),
                "min": round(rd_min, 2), "max": round(rd_max, 2),
                "valid_fraction": round(rd_frac, 3)
            },
            "disparity_difference_px": round(delta_d, 2),
            "disp_noise_std_px": round(disp_noise_std, 3),
            "statistically_distinguishable": bool(statistically_distinguishable),
            "expected_depth_diff_cm": round(expected_delta_z_cm, 2),
            "z_median_inside_m": round(in_z_med, 3),
            "z_median_road_m": round(rd_z_med, 3),
            "z_difference_cm": round(delta_z_cm, 2),
            "road_plane": {
                "equation": [round(float(v), 5) for v in plane_eq] if plane_eq else None,
                "inlier_ratio": round(inlier_ratio, 3) if plane_eq else None,
                "rmse_cm": round(rmse_m * 100.0, 2) if (plane_eq and rmse_m) else None,
                "normal_vertical_tilt_deg": round(vert_angle, 2) if (plane_eq and vert_angle) else None,
                "status": plane_status
            },
            "depth_metrology": {
                "mean_signed_depth_cm": round(signed_depth_cm, 2) if signed_depth_cm is not None else None,
                "p95_signed_depth_cm": round(p95_depth_cm, 2) if p95_depth_cm is not None else None,
                "max_signed_depth_cm": round(max_depth_cm, 2) if max_depth_cm is not None else None,
                "status": depth_status
            },
            "diagnostic_telemetry": {
                "y_inside_mean_m": round(y_inside_mean, 3) if y_inside_mean is not None else None,
                "y_road_mean_m": round(y_road_mean, 3) if y_road_mean is not None else None,
                "y_inside_gt_y_road": bool(y_inside_mean > y_road_mean) if (y_inside_mean and y_road_mean) else None
            },
            "area_metrology": {
                "shoelace_area_m2": round(area_m2, 4) if area_m2 is not None else None,
                "length_m": round(length_m, 2) if length_m is not None else None,
                "width_m": round(width_m, 2) if width_m is not None else None,
                "bbox_area_m2": round(bbox_area_m2, 4) if bbox_area_m2 is not None else None,
                "area_le_bbox_check": bool(area_m2 <= bbox_area_m2 * 1.05) if (area_m2 and bbox_area_m2) else None,
                "status": area_status if plane_eq else "NO_PLANE"
            }
        }
        audit_results.append(record)

        # Generate Visualizations for Diagnostic Images
        # Visual 1: Disparity inside vs outside
        disp_color = cv2.applyColorMap(np.clip(disp / 64.0 * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        disp_tile = left_rgb.copy()
        # Overlay pothole contour and road ring contour
        cv2.drawContours(disp_tile, [cnt], -1, (0, 0, 255), 2)
        rd_cnts, _ = cv2.findContours(road_ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp_tile, rd_cnts, -1, (255, 255, 0), 1)
        disp_vis_combined = np.hstack([cv2.resize(disp_tile, (480, 270)), cv2.resize(disp_color, (480, 270))])
        disp_vis_tiles.append(disp_vis_combined)

        # Visual 2: Reprojected 3D Polygon on RGB
        reproject_tile = left_rgb.copy()
        cv2.drawContours(reproject_tile, [cnt], -1, (0, 255, 0), 2) # Original green
        if polygon_reprojected_2d:
            poly_np = np.array(polygon_reprojected_2d, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(reproject_tile, [poly_np], isClosed=True, color=(0, 165, 255), thickness=2) # Reprojected orange
        cv2.putText(reproject_tile, f"Frame {f_idx}: Green=Visible, Orange=3D Plane Reproject",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        reproject_vis_tiles.append(cv2.resize(reproject_tile, (640, 360)))

    # Save diagnostic images
    if disp_vis_tiles:
        disp_audit_img = np.vstack(disp_vis_tiles)
        cv2.imwrite(str(eval_dir / "stereo_disparity_audit.jpg"), disp_audit_img)
        print(f"Saved disparity audit image to: {eval_dir / 'stereo_disparity_audit.jpg'}")

    if reproject_vis_tiles:
        reproject_audit_img = np.vstack(reproject_vis_tiles)
        cv2.imwrite(str(eval_dir / "pothole_3d_polygon_audit.jpg"), reproject_audit_img)
        print(f"Saved 3D polygon reprojection audit image to: {eval_dir / 'pothole_3d_polygon_audit.jpg'}")

    # Save JSON Diagnostic
    with open(eval_dir / "pothole_geometry_diagnostic.json", "w") as f:
        json.dump({
            "noise_floor_analysis": {
                "sample_pixels": len(flat_residuals),
                "disp_noise_std_px": round(disp_noise_std, 4),
                "disp_noise_iqr_px": round(disp_noise_iqr, 4),
                "distinguishability_threshold_1_96_sigma_px": round(1.96 * disp_noise_std, 4)
            },
            "potholes_audited": audit_results
        }, f, indent=2)
    print(f"Saved pothole geometry diagnostic JSON to: {eval_dir / 'pothole_geometry_diagnostic.json'}")

    # PRINT PART G CRITICAL DEPTH DIAGNOSTIC TABLE
    print("\n" + "=" * 95)
    print("PART G — CRITICAL DEPTH DIAGNOSTIC TABLE")
    print("=" * 95)
    headers = ["Fr", "ID", "d_in", "d_rd", "Δd (px)", "Noise σ", "Dist?", "ΔZ_th", "p95 Dep", "Max Dep", "Area m²", "Status"]
    row_fmt = "{:<4} {:<3} {:<6} {:<6} {:<8} {:<8} {:<6} {:<8} {:<8} {:<8} {:<8} {:<10}"
    print(row_fmt.format(*headers))
    print("-" * 95)
    for r in audit_results:
        p95_str = f"{r['depth_metrology']['p95_signed_depth_cm']:.1f}cm" if r['depth_metrology']['p95_signed_depth_cm'] is not None else "None"
        max_str = f"{r['depth_metrology']['max_signed_depth_cm']:.1f}cm" if r['depth_metrology']['max_signed_depth_cm'] is not None else "None"
        area_str = f"{r['area_metrology']['shoelace_area_m2']:.3f}" if r['area_metrology']['shoelace_area_m2'] is not None else "None"
        print(row_fmt.format(
            r["frame_index"],
            r["pothole_id"],
            r["disparity_inside"]["median"],
            r["disparity_road"]["median"],
            r["disparity_difference_px"],
            r["disp_noise_std_px"],
            "YES" if r["statistically_distinguishable"] else "NO",
            f"{r['expected_depth_diff_cm']:.1f}cm",
            p95_str,
            max_str,
            area_str,
            r["depth_metrology"]["status"]
        ))
    print("=" * 95)

    # ANSWER 5 CRITICAL QUESTIONS HONESTLY (Part G)
    print("\n--- ANSWERS TO THE 5 CRITICAL DIAGNOSTIC QUESTIONS (PART G) ---")
    for r in audit_results:
        f = r["frame_index"]
        print(f"\n[Frame {f:02d} Pothole ID {r['pothole_id']}]:")
        q1 = r["z_median_inside_m"] > r["z_median_road_m"]
        print(f"  Q1: Is Z_inside > Z_road? {'YES' if q1 else 'NO'} (Z_in={r['z_median_inside_m']}m vs Z_rd={r['z_median_road_m']}m, diff={r['z_difference_cm']:.1f}cm)")
        y_inside = r["diagnostic_telemetry"]["y_inside_mean_m"]
        y_road = r["diagnostic_telemetry"]["y_road_mean_m"]
        q2 = r["diagnostic_telemetry"]["y_inside_gt_y_road"]
        print(f"  Q2: Is Y_inside > Y_road? {'YES' if q2 else 'NO'} (Y_in={y_inside}m vs Y_rd={y_road}m) [Secondary Telemetry]")
        print(f"  Q3: What is the noise floor? sigma_d = {r['disp_noise_std_px']} px (Road plane RMSE = {r['road_plane']['rmse_cm']} cm)")
        q4 = r["statistically_distinguishable"]
        print(f"  Q4: Is measured depth larger than noise floor? {'YES' if q4 else 'NO'} (|Δd|={abs(r['disparity_difference_px'])} px vs 1.96*sigma={1.96*r['disp_noise_std_px']:.3f} px)")
        if not q4:
            print(f"  Q5: Cavity signal is WITHIN THE NOISE FLOOR. Stereo disparity does NOT statistically distinguish depression.")
        else:
            print(f"  Q5: Cavity signal is statistically distinguishable above the noise floor.")

    return audit_results

if __name__ == "__main__":
    run_stereo_disparity_audit()
