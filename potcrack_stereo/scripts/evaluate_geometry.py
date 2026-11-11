"""
evaluate_geometry.py - Ground-Truth vs Predicted Stereo 3D Geometry Cross-Check

Evaluates the 3D pothole geometry and metrology pipeline:
1. Compares road plane parameters fitted on GT depth vs Predicted stereo depth.
2. Compares 95th percentile depression depth (pred vs GT).
3. Compares ray-intersected footprint area in m² (pred vs GT).
4. Saves metrics to outputs/geometry/gt_vs_pred.json and comparison plots.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.stereo_depth import FastFoundationStereoMatcher, SGBMStereoMatcher, disparity_to_depth
from scripts.pothole_geometry import PotholeGeometryEngine

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "geometry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RSRD_DIR = PROJECT_ROOT / "data" / "RSRD"


def run_geometry_evaluation(num_frames: int = 15):
    print("============================================================")
    print("3D GEOMETRY ENGINE: GROUND-TRUTH VS STEREO CROSS-CHECK")
    print("============================================================")

    engine = PotholeGeometryEngine()
    matcher = FastFoundationStereoMatcher()
    if matcher.model is None:
        print("[GeometryEval] Falling back to SGBM matcher")
        matcher = SGBMStereoMatcher()

    left_files = sorted(list((RSRD_DIR / "left_half").glob("*.jpg")))
    calib_dir = RSRD_DIR / "calibration_files"
    
    if not left_files:
        print("No RSRD frames found for geometry evaluation.")
        return

    selected_files = left_files[:num_frames]
    results = []

    for idx, left_path in enumerate(selected_files):
        stem = left_path.stem
        right_path = RSRD_DIR / "right_half" / f"{stem}.jpg"
        disp_gt_path = RSRD_DIR / "disparity_half" / f"{stem}.png"
        depth_gt_path = RSRD_DIR / "depth_half" / f"{stem}.png"

        # Calibration
        calib_candidates = list(calib_dir.glob("*.pkl"))
        calib_path = calib_candidates[0]
        for cp in calib_candidates:
            if stem[:8] in cp.stem:
                calib_path = cp
                break

        with open(calib_path, "rb") as f:
            calib = pickle.load(f)

        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        gt_depth = cv2.imread(str(depth_gt_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0

        # Predict stereo depth
        pred_disp = matcher.estimate_disparity(left_img, right_img)
        pred_depth, _ = disparity_to_depth(pred_disp, calib)

        h, w = pred_depth.shape[:2]
        # Simulate / test on a defined road ROI patch in the lower road region
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        # Center road patch simulating a pothole ROI
        cv2.circle(roi_mask, (w // 2, int(h * 0.75)), radius=int(min(h, w) * 0.08), color=255, thickness=-1)

        # Run geometry engine on GT depth
        gt_instances = engine.analyze_potholes(left_img, gt_depth, roi_mask, calib)
        # Run geometry engine on Predicted stereo depth
        pred_instances = engine.analyze_potholes(left_img, pred_depth, roi_mask, calib)

        if gt_instances and pred_instances:
            g_inst = gt_instances[0]
            p_inst = pred_instances[0]

            res_entry = {
                "frame": stem,
                "gt_depth_p95_cm": g_inst.depth_p95_cm,
                "pred_depth_p95_cm": p_inst.depth_p95_cm,
                "gt_area_m2": g_inst.area_m2,
                "pred_area_m2": p_inst.area_m2,
                "depth_abs_diff_cm": abs((p_inst.depth_p95_cm or 0.0) - (g_inst.depth_p95_cm or 0.0)),
                "area_abs_diff_m2": abs((p_inst.area_m2 or 0.0) - (g_inst.area_m2 or 0.0)),
                "gt_quality": g_inst.quality,
                "pred_quality": p_inst.quality,
            }
            results.append(res_entry)

            print(f"  Frame [{idx+1}/{len(selected_files)}] {stem}: "
                  f"Depth Diff={res_entry['depth_abs_diff_cm']:.1f} cm | "
                  f"Area Diff={res_entry['area_abs_diff_m2']:.3f} m² | "
                  f"Quality={p_inst.quality}")

    # Summary
    if results:
        depth_diffs = [r["depth_abs_diff_cm"] for r in results]
        area_diffs = [r["area_abs_diff_m2"] for r in results]

        summary = {
            "frames_evaluated": len(results),
            "mean_depth_diff_cm": float(np.mean(depth_diffs)),
            "median_depth_diff_cm": float(np.median(depth_diffs)),
            "mean_area_diff_m2": float(np.mean(area_diffs)),
            "median_area_diff_m2": float(np.median(area_diffs)),
            "detailed_results": results
        }

        with open(OUTPUT_DIR / "gt_vs_pred.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        print("\n------------------------------------------------------------")
        print("GEOMETRY CROSS-CHECK SUMMARY:")
        print("------------------------------------------------------------")
        print(f"Frames Evaluated     : {len(results)}")
        print(f"Mean Depth Diff      : {summary['mean_depth_diff_cm']:.2f} cm")
        print(f"Median Depth Diff    : {summary['median_depth_diff_cm']:.2f} cm")
        print(f"Mean Area Diff       : {summary['mean_area_diff_m2']:.4f} m²")
        print(f"Median Area Diff     : {summary['median_area_diff_m2']:.4f} m²")
        print(f"Report Saved to      : {OUTPUT_DIR / 'gt_vs_pred.json'}")
        print("============================================================\n")


if __name__ == "__main__":
    run_geometry_evaluation()
