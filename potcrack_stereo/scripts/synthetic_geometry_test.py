#!/usr/bin/env python3
"""
scripts/synthetic_geometry_test.py - Synthetic Ground-Truth Geometry Test
Verifies the exact mathematics of PotholeGeometryEngine:
- Road plane fitting
- Ray/plane intersection
- 3D Shoelace footprint area
- Signed perpendicular depression depth
using a known synthetic 3D ground truth scene (no stereo noise).
"""

import sys
import math
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pothole_geometry import PotholeGeometryEngine

def run_synthetic_test():
    print("=" * 75)
    print(" BUSSENSE PHYSICAL CALIBRATION & SYNTHETIC GEOMETRY AUDIT")
    print("=" * 75)

    # 1. Camera Intrinsics
    W, H = 1920, 1080
    fx, fy = 1050.0, 1050.0
    cx, cy = 960.0, 540.0
    baseline = 0.12
    calib = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline}

    # 2. Synthetic Road Plane
    # Camera mounted at height h_cam = 1.25 m, pitched downward by pitch_deg = 8 degrees.
    # In camera frame: X=right, Y=down, Z=forward.
    # Pitch down means optical axis points slightly below horizontal.
    # Normal pointing UP out of the road:
    pitch_rad = math.radians(8.0)
    # Unit normal pointing up:
    ny = -math.cos(pitch_rad) # Y is down, so -cos(pitch) points UP
    nz = -math.sin(pitch_rad) # optical axis points slightly down, so road climbs in Z relative to axis: nz = -sin(pitch)
    nx = 0.0
    norm_vec = np.array([nx, ny, nz], dtype=np.float64)
    norm_vec /= np.linalg.norm(norm_vec)
    # Plane: aX + bY + cZ + d = 0
    # At (X=0, Z=0), camera is at Y=0, road is at Y = +h_cam = 1.25 m.
    # So point on road: (0, h_cam, 0)
    # nx*0 + ny*h_cam + nz*0 + d = 0 => d = -ny * h_cam
    h_cam = 1.25
    d_true = -float(norm_vec[1] * h_cam)
    true_plane = (float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]), float(d_true))
    print(f"Ground Truth Road Plane: {true_plane[0]:.4f}X + {true_plane[1]:.4f}Y + {true_plane[2]:.4f}Z + {true_plane[3]:.4f} = 0")
    print(f"  Camera height: {h_cam:.2f} m, Pitch: 8.0°")

    # 3. Define Known Pothole on the Road
    # Centered at Z_center = 4.5 m, X_center = 0.1 m
    # Dimensions: Length (along road) = 0.60 m, Width (across road) = 0.40 m
    # Known True Area = 0.60 * 0.40 = 0.240 m²
    true_length = 0.60
    true_width = 0.40
    true_area = true_length * true_width # 0.240 m²
    true_depth_cm = 5.0 # 5.0 cm depression (0.05 m below road plane)

    z_c = 4.5
    x_c = 0.1
    # Y on road plane: a*x + b*y + c*z + d = 0 => y = -(a*x + c*z + d) / b
    y_c = -(norm_vec[0] * x_c + norm_vec[2] * z_c + d_true) / norm_vec[1]
    print(f"\nKnown Pothole Ground Truth:")
    print(f"  Center: ({x_c:.2f}m, {y_c:.2f}m, {z_c:.2f}m)")
    print(f"  True Dimensions: Length = {true_length:.3f} m, Width = {true_width:.3f} m")
    print(f"  True Footprint Area: {true_area:.4f} m²")
    print(f"  True Depth Depression: {true_depth_cm:.1f} cm")

    # 4. Generate image grid and synthetic depth map
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    
    # Rays:
    rx = (u_grid - cx) / fx
    ry = (v_grid - cy) / fy
    rz = np.ones_like(rx)

    # Ray-road plane intersection depth Z:
    # (nx*rx + ny*ry + nz*rz)*Z + d = 0 => Z = -d / (nx*rx + ny*ry + nz)
    denom = norm_vec[0] * rx + norm_vec[1] * ry + norm_vec[2] * rz
    road_z = np.where(denom < -1e-4, -d_true / denom, 0.0).astype(np.float32)

    # 3D coordinates on road
    X_road = rx * road_z
    Y_road = ry * road_z
    Z_road = road_z

    # Check if point is inside the pothole rectangle
    # Along road direction vector (unit vector parallel to road in Z-direction):
    # Road normal is (0, ny, nz). A vector along the road heading forward:
    v_fwd = np.array([0.0, -norm_vec[2], norm_vec[1]]) # perp to norm_vec
    v_fwd /= np.linalg.norm(v_fwd)
    v_right = np.array([1.0, 0.0, 0.0]) # lateral across road

    # Delta from center:
    dX = X_road - x_c
    dY = Y_road - y_c
    dZ = Z_road - z_c

    # Project delta onto road longitudinal and lateral axes:
    long_dist = dY * v_fwd[1] + dZ * v_fwd[2]
    lat_dist = dX * v_right[0]

    pothole_mask = (
        (np.abs(long_dist) <= (true_length / 2.0)) &
        (np.abs(lat_dist) <= (true_width / 2.0)) &
        (road_z > 1.0)
    ).astype(np.uint8) * 255

    n_pothole_pixels = int(np.sum(pothole_mask > 0))
    print(f"  Projected Pothole Mask Pixels: {n_pothole_pixels}")

    # Construct synthetic depth map:
    # Outside pothole: Z = road_z
    # Inside pothole: ray intersects floor plane at depth D_perp below road:
    # Z_cavity = (-d_true - D_perp) / denom
    depth_map = road_z.copy()
    d_perp_m = float(true_depth_cm / 100.0)
    z_cavity = np.where(denom < -1e-4, (-d_true - d_perp_m) / denom, road_z).astype(np.float32)
    depth_map[pothole_mask > 0] = z_cavity[pothole_mask > 0]

    # 5. Run PotholeGeometryEngine on the Synthetic Scene
    engine = PotholeGeometryEngine()
    
    # Create fake RGB image
    left_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    instances = engine.analyze_potholes(left_rgb, depth_map, pothole_mask, calib)

    print("\n" + "-" * 75)
    print(" PotholeGeometryEngine Reconstruction Results:")
    print("-" * 75)
    assert len(instances) == 1, f"Expected 1 instance, got {len(instances)}"
    inst = instances[0]

    print(f"  Reconstructed Quality     : {inst.quality} ({inst.quality_reason})")
    print(f"  Fitted Plane Normal       : {inst.plane_equation[:3] if inst.plane_equation else None}")
    print(f"  True Plane Normal         : {true_plane[:3]}")
    print(f"  Plane Tilt Angle          : {inst.plane_vertical_angle_deg:.2f}° (Expected: 8.0°)")
    print(f"  Plane RMSE                : {inst.plane_rmse_m*100.0 if inst.plane_rmse_m else 0.0:.3f} cm")
    print(f"  Measured Depth (p95)      : {inst.depth_p95_cm} cm (True: {true_depth_cm:.1f} cm)")
    print(f"  Measured Depth (Median)   : {inst.depth_median_cm} cm (True: {true_depth_cm:.1f} cm)")
    print(f"  Measured Depth (Max/p99)  : {inst.depth_max_cm} cm (True: {true_depth_cm:.1f} cm)")
    print(f"  Measured Area             : {inst.area_m2} m² (True: {true_area:.4f} m²)")
    print(f"  Measured Dimensions       : {inst.length_m} m x {inst.width_m} m (True: {true_length:.2f}m x {true_width:.2f}m)")

    # 6. Mathematical Error Analysis
    depth_err = abs(inst.depth_p95_cm - true_depth_cm) if inst.depth_p95_cm else 999.0
    area_err_pct = abs(inst.area_m2 - true_area) / true_area * 100.0 if inst.area_m2 else 999.0

    print(f"\nDiscrepancy:")
    print(f"  Depth Error               : {depth_err:.2f} cm")
    print(f"  Area Error                : {area_err_pct:.2f}%")

    # Sanity check: area <= length * width
    if inst.area_m2 and inst.length_m and inst.width_m:
        bounding_area = inst.length_m * inst.width_m
        print(f"  Bounding Area (L x W)     : {bounding_area:.4f} m² >= Footprint Area {inst.area_m2:.4f} m²: {bounding_area >= inst.area_m2 * 0.99}")

    print("=" * 75)
    return inst

if __name__ == "__main__":
    run_synthetic_test()
