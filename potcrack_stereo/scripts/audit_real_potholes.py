#!/usr/bin/env python3
"""
scripts/audit_real_potholes.py - Complete Audit of Real Pothole Depth and Area
Inspects:
- Image resolution
- Calibration: fx, fy, cx, cy, baseline
- Disparity map inside vs outside pothole
- SGBM disparity resolution / step size
- What delta Z corresponds to 1 px disparity: dZ = (Z^2 / (f*B)) * d(disp)
- 3D points distribution inside pothole vs surrounding ring
- Fitted plane equation, normal, tilt, RMSE
- Signed depression percentiles: min, median, p90, p95, p99, max
- Ray intersection points and polygon area
"""

import sys
import math
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stereo_depth import get_stereo_matcher
from scripts.pothole_geometry import PotholeGeometryEngine
from ultralytics import YOLO

def audit_potholes():
    data_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
    left_files = sorted(list((data_dir / "left").glob("*.jpg")))
    right_files = sorted(list((data_dir / "right").glob("*.jpg")))

    calib = {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 960.0,
        "cy": 540.0,
        "baseline": 0.12,
        "image_width": 1920,
        "image_height": 1080
    }
    fx, fy, cx, cy, B = calib["fx"], calib["fy"], calib["cx"], calib["cy"], calib["baseline"]

    matcher = get_stereo_matcher("sgbm")
    yolo = YOLO("checkpoints/segmentation/best_pothole_seg.pt")
    engine = PotholeGeometryEngine()

    test_frames = [5, 15, 25, 35]

    for f_idx in test_frames:
        if f_idx >= len(left_files):
            continue
        left_img = cv2.imread(str(left_files[f_idx]))
        right_img = cv2.imread(str(right_files[f_idx]))
        H, W = left_img.shape[:2]

        print("\n" + "=" * 80)
        print(f" AUDIT FOR FRAME {f_idx:03d} (Resolution: {W}x{H})")
        print("=" * 80)

        # 1. SGBM Disparity
        disp = matcher.estimate_disparity(left_img, right_img)
        valid_disp = (disp > 0.5) & np.isfinite(disp)
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[valid_disp] = (fx * B) / disp[valid_disp]

        print(f"Global Disparity Stats:")
        print(f"  Valid fraction: {np.mean(valid_disp):.1%}")
        if np.any(valid_disp):
            print(f"  Min/Med/Max disp: {np.min(disp[valid_disp]):.2f} / {np.median(disp[valid_disp]):.2f} / {np.max(disp[valid_disp]):.2f} px")
            print(f"  Min/Med/Max depth: {np.min(depth[valid_disp]):.2f} / {np.median(depth[valid_disp]):.2f} / {np.max(depth[valid_disp]):.2f} m")

        # 2. YOLO Segmentation
        yolo_res = yolo(left_img, conf=0.20, verbose=False)[0]
        n_dets = len(yolo_res.boxes) if yolo_res.boxes is not None else 0
        print(f"\nYOLO Detections: {n_dets} boxes")

        if n_dets == 0:
            continue

        for i, (box, m) in enumerate(zip(yolo_res.boxes, yolo_res.masks.data)):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            m_np = m.cpu().numpy()
            m_orig = cv2.resize((m_np > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) * 255
            
            pothole_px = int(np.sum(m_orig > 0))
            box_w = xyxy[2] - xyxy[0]
            box_h = xyxy[3] - xyxy[1]
            print(f"\n--- Detection #{i+1} (Conf: {conf:.2f}, bbox: [{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}], size: {box_w}x{box_h} px, mask area: {pothole_px} px) ---")

            # Extract ring
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (engine.ring_dilation_px, engine.ring_dilation_px))
            dilated = cv2.dilate(m_orig, kernel, iterations=1)
            ring = cv2.bitwise_and(dilated, cv2.bitwise_not(m_orig))

            # Disparity and depth inside pothole vs ring
            disp_in_pot = disp[(m_orig > 0) & valid_disp]
            disp_in_ring = disp[(ring > 0) & valid_disp]
            depth_in_pot = depth[(m_orig > 0) & valid_disp]
            depth_in_ring = depth[(ring > 0) & valid_disp]

            print(f"  Pothole interior: {len(disp_in_pot)}/{pothole_px} valid stereo pixels ({len(disp_in_pot)/max(1, pothole_px):.1%})")
            if len(disp_in_pot) > 0:
                print(f"    Interior Disp : min={np.min(disp_in_pot):.2f}, med={np.median(disp_in_pot):.2f}, max={np.max(disp_in_pot):.2f} px")
                print(f"    Interior Depth: min={np.min(depth_in_pot):.2f}m, med={np.median(depth_in_pot):.2f}m, max={np.max(depth_in_pot):.2f}m")
            print(f"  Ring surrounding: {len(disp_in_ring)} valid stereo pixels")
            if len(disp_in_ring) > 0:
                print(f"    Ring Disp     : min={np.min(disp_in_ring):.2f}, med={np.median(disp_in_ring):.2f}, max={np.max(disp_in_ring):.2f} px")
                print(f"    Ring Depth    : min={np.min(depth_in_ring):.2f}m, med={np.median(depth_in_ring):.2f}m, max={np.max(depth_in_ring):.2f}m")

            # Check theoretical depth resolution at this distance
            if len(depth_in_pot) > 0:
                z_med = float(np.median(depth_in_pot))
                # SGBM subpixel step size is 1/16 pixel = 0.0625 px
                # dZ / d(disp) = Z^2 / (fx * B)
                # For 1 full pixel:
                delta_z_1px = (z_med ** 2) / (fx * B)
                delta_z_subpx = delta_z_1px / 16.0 # 1/16 px
                print(f"  Theoretical Depth Resolution at Z={z_med:.2f}m (fx={fx}, B={B}m):")
                print(f"    1.0 px disparity shift = {delta_z_1px*100:.2f} cm")
                print(f"    1/16 px subpixel shift = {delta_z_subpx*100:.2f} cm")

            # Plane Fit
            plane_eq, inlier_ratio, rmse_m, vert_angle, plane_msg = engine.fit_road_plane(depth, m_orig, fx, fy, cx, cy)
            print(f"  Road Plane Fit: status='{plane_msg}'")
            if plane_eq is not None:
                a, b, c, d = plane_eq
                print(f"    Equation: {a:.4f}X + {b:.4f}Y + {c:.4f}Z + {d:.4f} = 0")
                print(f"    Tilt: {vert_angle:.1f}°, Inlier ratio: {inlier_ratio:.1%}, RMSE: {rmse_m*100:.2f} cm")

                # Signed Depression
                res_dep = engine.compute_signed_depth(depth, m_orig, plane_eq, fx, fy, cx, cy)
                d_p95, d_med, d_max, d_p99, raw_max_cm, center_3d, total_px, valid_px, valid_frac, dep_msg = res_dep
                print(f"  Signed Depression: status='{dep_msg}'")
                print(f"    Percentiles: med={d_med}, p95={d_p95}, p99={d_p99}, raw_max={raw_max_cm} cm")

                # Contours & Area
                contours, _ = cv2.findContours(m_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    res_area = engine.compute_footprint_area_rays(cnt, plane_eq, fx, fy, cx, cy)
                    area_m2, len_m, wid_m, raw_area_m2, area_msg = res_area
                    print(f"  Footprint Area: status='{area_msg}'")
                    print(f"    Area={area_m2} m² (raw={raw_area_m2:.4f} m²), Dims={len_m}m x {wid_m}m")
                    if len_m and wid_m and area_m2:
                        print(f"    Sanity check (Area <= L*W): {area_m2:.4f} <= {len_m*wid_m:.4f} m² -> {area_m2 <= len_m*wid_m * 1.01}")

if __name__ == "__main__":
    audit_potholes()
