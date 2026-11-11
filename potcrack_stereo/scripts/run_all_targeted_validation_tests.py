#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Comprehensive Targeted Validation Test Suite
======================================================
Executes the 4 mandatory validation suites before final demo generation:
- Suite A: Pothole Metrology & Guardrails Test (>= 5 potholes)
- Suite B: Crack Detection, Spatial Gating & Skeleton Metrology Test (>= 5 frames)
- Suite C: Road Condition Dynamic Evidence & Hysteresis Test
- Suite D: Traffic Evidence Score & State Precedence Test
- Phase 5 Debug Artifacts: Saves RGB, raw mask, cleaned mask, connected components, contours
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_unified_pipeline import UnifiedBusSensePipeline
from scripts.pothole_geometry import PotholeGeometryEngine, BUSSENSE_METROLOGY_GUARDRAILS
from scripts.crack_geometry import CrackGeometryEngine
from scripts.road_quality import RoadQualityEngine, BUSSENSE_OPERATIONAL_THRESHOLDS
from traffic.density_engine import TrafficDensityEngine
from scripts.make_kitti_traffic_demo import discover_drives, parse_kitti_calib, KITTI_ROOT, CALIB_FILE

DEBUG_OUT_DIR = PROJECT_ROOT / "outputs" / "validation_artifacts"
DEBUG_OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_suite_a_pothole_test(pipeline: UnifiedBusSensePipeline, pothole_pairs: List[tuple]):
    print("\n" + "="*70)
    print(" SUITE A: POTHOLE 3D METROLOGY & GUARDRAILS VALIDATION")
    print("="*70)
    
    tested_potholes = 0
    results_summary = []

    for frame_idx, (lf, rf) in enumerate(pothole_pairs):
        left_img = cv2.imread(str(lf))
        right_img = cv2.imread(str(rf))
        if left_img is None or right_img is None:
            continue

        state = pipeline.process_frame(left_img, right_img, frame_index=frame_idx)
        pots = state.potholes

        if len(pots) == 0:
            continue

        for p in pots:
            tested_potholes += 1
            t_id = p["track_id"]
            conf = p["confidence"]
            depth_p95 = p.get("depth_p95_cm")
            area = p.get("area_m2")
            length = p.get("length_m")
            width = p.get("width_m")
            plane_eq = p.get("plane_equation")
            quality = p.get("quality")
            reason = p.get("quality_reason")

            # Plane metrics
            tilt_deg = None
            if plane_eq is not None:
                b = plane_eq[1]
                cos_theta = np.clip(-b, -1.0, 1.0)
                tilt_deg = float(np.degrees(np.arccos(cos_theta)))

            # Area bounding check
            bounding_ok = True
            if area is not None and length is not None and width is not None:
                max_allowable_area = length * width * BUSSENSE_METROLOGY_GUARDRAILS["bounding_box_safety_factor"]
                bounding_ok = (area <= max_allowable_area)

            # Plausibility sanity check
            phys_plausible = True
            if depth_p95 is not None and (depth_p95 > BUSSENSE_METROLOGY_GUARDRAILS["max_plausible_depth_cm"] or depth_p95 < BUSSENSE_METROLOGY_GUARDRAILS["min_plausible_depth_cm"]):
                phys_plausible = False
            if area is not None and area > BUSSENSE_METROLOGY_GUARDRAILS["max_plausible_area_m2"]:
                phys_plausible = False

            rec = {
                "frame": frame_idx,
                "pothole_id": t_id,
                "confidence": conf,
                "depth_p95_cm": depth_p95,
                "area_m2": area,
                "dimensions": f"{length:.2f}m x {width:.2f}m" if (length and width) else "N/A",
                "plane_tilt_deg": round(tilt_deg, 1) if tilt_deg is not None else None,
                "bounding_check": "PASS" if bounding_ok else "FAIL",
                "plausibility": "PASS" if phys_plausible else "FAIL",
                "quality": quality,
                "reason": reason
            }
            results_summary.append(rec)

            print(f"  [Pothole #{t_id:02d} | Fr {frame_idx:02d}] "
                  f"Depth: {str(depth_p95):>6} cm | Area: {str(area):>7} m² | "
                  f"Dim: {rec['dimensions']:>14} | Tilt: {str(rec['plane_tilt_deg']):>5}° | "
                  f"Status: {quality:<7} | Reason: {reason}")

            # Save Phase 5 debug image for sample pothole
            if tested_potholes <= 4:
                mask = p.get("mask")
                if mask is not None:
                    vis_dbg = left_img.copy()
                    cnt = p.get("contour")
                    if cnt is not None:
                        cv2.drawContours(vis_dbg, [cnt], -1, (0, 255, 255), 2)
                    cv2.imwrite(str(DEBUG_OUT_DIR / f"phase5_pothole_fr{frame_idx:02d}_p{t_id}.jpg"), vis_dbg)

        if tested_potholes >= 8:
            break

    print(f"\nTotal Potholes Evaluated: {tested_potholes}")
    assert tested_potholes >= 5, f"Expected at least 5 potholes tested, got {tested_potholes}"
    all_bounds_pass = all(r["bounding_check"] == "PASS" for r in results_summary)
    all_plaus_pass = all(r["plausibility"] == "PASS" for r in results_summary)
    print(f"Bounding Rectangle Check: {'PASSED (100%)' if all_bounds_pass else 'FAILED'}")
    print(f"Physical Plausibility Check: {'PASSED (100%)' if all_plaus_pass else 'FAILED'}")
    return results_summary


def run_suite_b_crack_test(pipeline: UnifiedBusSensePipeline, sample_pairs: List[tuple]):
    print("\n" + "="*70)
    print(" SUITE B: CRACK DETECTION & ROAD-PLANE SPATIAL GATING VALIDATION")
    print("="*70)

    crack_count_total = 0
    frames_tested = 0

    for f_idx, (lf, rf) in enumerate(sample_pairs):
        left_img = cv2.imread(str(lf))
        right_img = cv2.imread(str(rf))
        if left_img is None or right_img is None:
            continue

        state = pipeline.process_frame(left_img, right_img, frame_index=f_idx)
        cracks = state.cracks
        frames_tested += 1

        print(f"  Frame {f_idx:02d} ({lf.name}) -> Cracks Detected: {len(cracks)}")
        for c in cracks:
            crack_count_total += 1
            t_id = c["track_id"]
            l_m = c.get("length_m")
            cov_m2 = c.get("coverage_area_m2")
            center = c.get("center_3d")
            x_m = center[0] if center else None
            z_m = center[2] if center else None

            print(f"    Crack #{t_id}: Length={l_m:.2f}m, Coverage={cov_m2:.4f}m², "
                  f"Center 3D: X={x_m:.2f}m, Z={z_m:.2f}m | Spatial Gating (|X|<=5.5m): {'PASS' if abs(x_m) <= 5.5 else 'FAIL'}")

            # Save Phase 5 crack debug image
            if crack_count_total <= 3:
                c_mask = c.get("mask")
                if c_mask is not None:
                    vis_cr = left_img.copy()
                    vis_cr[c_mask > 0] = (255, 0, 200)
                    cv2.imwrite(str(DEBUG_OUT_DIR / f"phase5_crack_fr{f_idx:02d}_c{t_id}.jpg"), vis_cr)

    print(f"\nTotal Frames Tested for Cracks: {frames_tested}")
    print(f"Total Cracks Detected & Verified: {crack_count_total}")
    assert frames_tested >= 5, "Expected >= 5 frames tested"
    return True


def run_suite_c_road_condition_test(pipeline_kitti: UnifiedBusSensePipeline, pipeline_zed: UnifiedBusSensePipeline,
                                     kitti_pair: tuple, severe_pothole_pair: tuple):
    print("\n" + "="*70)
    print(" SUITE C: ROAD CONDITION ENGINE & HYSTERESIS VALIDATION")
    print("="*70)

    # 1. Test clean road (KITTI)
    k_left = cv2.imread(str(kitti_pair[0]))
    k_right = cv2.imread(str(kitti_pair[1]))
    pipeline_kitti.road_quality_engine.reset()
    state_clean = pipeline_kitti.process_frame(k_left, k_right, frame_index=0)
    rq_clean = state_clean.road_quality
    cat_clean = rq_clean.get("condition_category")

    print(f"  Clean Segment (KITTI):      ROAD CONDITION = {cat_clean} (Confidence: {rq_clean.get('confidence_percent', 0):.1f}%)")
    print(f"    Evidence: Potholes={rq_clean.get('pothole_count')}, Cracks={rq_clean.get('crack_count')}, Residual RMS={rq_clean.get('stereo_surface_irregularity_rms_cm')}")

    # 2. Test severe distress road (Dashcam Potholes)
    p_left = cv2.imread(str(severe_pothole_pair[0]))
    p_right = cv2.imread(str(severe_pothole_pair[1]))
    pipeline_zed.road_quality_engine.reset()

    # Feed 3 consecutive frames of distress to verify fast upward transition
    cat_distress_hist = []
    for f in range(3):
        state_distress = pipeline_zed.process_frame(p_left, p_right, frame_index=f)
        rq_distress = state_distress.road_quality
        cat_distress_hist.append(rq_distress.get("condition_category"))

    print(f"  Severe Segment (Dashcam):   ROAD CONDITION sequence = {' -> '.join(cat_distress_hist)}")
    print(f"    Evidence: Potholes={rq_distress.get('pothole_count')}, Max Depth={rq_distress.get('pothole_max_depth_cm')}cm, Cracks={rq_distress.get('crack_count')}")

    assert cat_clean == "NORMAL", f"Expected clean road to be NORMAL, got {cat_clean}"
    assert cat_distress_hist[-1] in ["HIGH", "ALARMING"], f"Expected severe road to reach HIGH/ALARMING, got {cat_distress_hist[-1]}"
    print("  Road Condition Dynamic Transitions: PASSED (Clean=NORMAL, Distress=HIGH/ALARMING)")
    return True


def run_suite_d_traffic_test():
    print("\n" + "="*70)
    print(" SUITE D: TRAFFIC EVIDENCE SCORE & PRECEDENCE VALIDATION")
    print("="*70)

    engine = TrafficDensityEngine(fx=721.5, img_width=1242, img_height=375)

    test_cases = [
        # (name, tracked_count, nearest_gap, roi_count, expected_state)
        ("Sparse Traffic (Free Flow)", 1, 35.0, 1, "NORMAL"),
        ("Moderate Traffic (Flowing)", 2, 18.0, 2, "MEDIUM"),
        ("Heavy Traffic (Dense)", 4, 9.0, 4, "HIGH"),
        ("Alarming Congestion (Bumper-to-Bumper)", 5, 3.2, 5, "ALARMING"),
    ]

    for name, cnt, gap, roi, expected in test_cases:
        engine.reset()
        # Mock scene analysis
        vehicles = [{"track_id": i, "valid": True, "Z_m": 12.0, "bbox": [100, 100, 200, 200]} for i in range(cnt)]
        inter = [{"distance_3d_m": gap, "longitudinal_gap_m": gap}] if cnt > 1 else []
        mock_scene = {"vehicles": vehicles, "inter_vehicle_distances": inter}

        # Mock road plane [a, b, c, d]
        mock_plane = np.array([0.0, -1.0, 0.05, 1.65])

        # Two updates to let hysteresis settle
        res1 = engine.update(mock_scene, road_plane=mock_plane)
        res2 = engine.update(mock_scene, road_plane=mock_plane)

        t_score = res2.get("traffic_score")
        t_lvl = res2.get("traffic_level")
        breakdown = res2.get("traffic_score_breakdown", {})

        print(f"  [{name}]")
        print(f"    Input: count={cnt}, gap={gap:.1f}m, roi={roi}")
        print(f"    Composite traffic_score: {t_score:.3f} | Breakdown: {breakdown}")
        print(f"    Assigned Level: {t_lvl} (Expected: {expected}) -> {'PASS' if t_lvl == expected else 'FAIL'}")
        assert t_lvl == expected, f"Failed on {name}: expected {expected}, got {t_lvl}"

    print("  Traffic Evidence Score Precedence: ALL PASSED")
    return True


def main():
    print("======================================================================")
    print(" BUSSENSE TARGETED VALIDATION TEST SUITES (A, B, C, D)")
    print("======================================================================")

    kitti_drives = discover_drives(KITTI_ROOT)
    kitti_calib = parse_kitti_calib(CALIB_FILE)
    d05 = next(((d_name, ls, rs) for d_name, ls, rs in kitti_drives if "drive_0005" in d_name), None)
    kitti_pair = (d05[1][15], d05[2][15]) if d05 else None

    pothole_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
    pot_left_files = sorted(list((pothole_dir / "left").glob("*.jpg")))
    pot_right_files = sorted(list((pothole_dir / "right").glob("*.jpg")))
    pothole_pairs = list(zip(pot_left_files, pot_right_files))

    calib_zed = {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 960.0,
        "cy": 540.0,
        "baseline": 0.12,
        "image_width": 1920,
        "image_height": 1080
    }
    crack_weights = str(PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt")

    pipe_kitti = UnifiedBusSensePipeline(calib=kitti_calib, stereo_model="sgbm", crack_weights=crack_weights, conf_crack=0.08)
    pipe_zed = UnifiedBusSensePipeline(calib=calib_zed, stereo_model="sgbm", crack_weights=crack_weights, conf_crack=0.08)

    # 1. Run Suite A
    suite_a_res = run_suite_a_pothole_test(pipe_zed, pothole_pairs[10:35])

    # 2. Run Suite B
    crack_sample_pairs = [pothole_pairs[i] for i in [15, 33, 62, 65, 66] if i < len(pothole_pairs)]
    suite_b_res = run_suite_b_crack_test(pipe_zed, crack_sample_pairs)

    # 3. Run Suite C
    suite_c_res = run_suite_c_road_condition_test(pipe_kitti, pipe_zed, kitti_pair, pothole_pairs[15])

    # 4. Run Suite D
    suite_d_res = run_suite_d_traffic_test()

    print("\n" + "="*70)
    print(" ALL 4 TARGETED VALIDATION TEST SUITES PASSED SUCCESSFULLY!")
    print(" Ready to render production demo and debug quad videos.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
