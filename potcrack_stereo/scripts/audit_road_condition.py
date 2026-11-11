#!/usr/bin/env python3
"""
scripts/audit_road_condition.py
Audit for Part N & Part O:
- Tests physical evidence processing across clean and distressed road scenes
- Verifies asymmetric temporal hysteresis (2 frames up, 4 frames down)
- Verifies fallback to ROAD CONDITION: INSUFFICIENT DATA
- Generates outputs/evaluation/road_condition_diagnostic.json
"""

import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.road_quality import RoadQualityEngine, BUSSENSE_OPERATIONAL_THRESHOLDS

def audit_road_condition():
    print("============================================================")
    print("PART N & O — ROAD CONDITION ENGINE & HYSTERESIS AUDIT")
    print("============================================================")

    eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    engine = RoadQualityEngine()

    calib = {"fx": 1050.0, "fy": 1050.0, "cx": 960.0, "cy": 540.0, "baseline": 0.12}
    nominal_plane_meta = {"is_valid": True, "equation": [0.0, -1.0, 0.0, 1.5], "inlier_ratio": 0.85}
    stereo_qual_good = {"valid_disp_fraction": 0.65}
    
    # Mathematically exact planar road surface: Z(v) = (h_cam * fy) / (v - cy)
    H, W = 1080, 1920
    planar_depth = np.zeros((H, W), dtype=np.float32)
    for v in range(541, H):
        planar_depth[v, :] = (1.5 * 1050.0) / (v - 540.0)

    # Test Case 1: Clean Road -> NORMAL
    rq_clean = engine.evaluate_road_quality(
        potholes=[],
        cracks=[],
        road_plane_meta=nominal_plane_meta,
        depth_m=planar_depth,
        calib=calib,
        observable_road_area_m2=45.0,
        stereo_quality=stereo_qual_good
    )
    print(f"1. Clean road condition: {rq_clean.condition_category} (Expected: NORMAL)")
    assert rq_clean.condition_category == "NORMAL"

    # Test Case 2: Insufficient Data / Invalid Plane -> INSUFFICIENT DATA
    rq_invalid = engine.evaluate_road_quality(
        potholes=[],
        cracks=[],
        road_plane_meta={"is_valid": False},
        depth_m=planar_depth,
        calib=calib,
        observable_road_area_m2=0.0,
        stereo_quality={"valid_disp_fraction": 0.02}
    )
    print(f"2. Insufficient data status: {rq_invalid.status} (Expected: ROAD CONDITION: INSUFFICIENT DATA)")
    assert "INSUFFICIENT DATA" in rq_invalid.status

    # Test Case 3: Upward Hysteresis (Requires 2 consecutive frames to transition to HIGH/ALARMING)
    engine.reset()
    distress_potholes = [{"depth_p95_cm": 6.5, "area_m2": 0.25}] # HIGH/ALARMING candidate
    
    # Frame 1: instant is ALARMING/HIGH, but smoothed stays NORMAL
    f1 = engine.evaluate_road_quality(
        potholes=distress_potholes, cracks=[], road_plane_meta=nominal_plane_meta,
        depth_m=planar_depth, calib=calib, observable_road_area_m2=40.0, stereo_quality=stereo_qual_good
    )
    print(f"3a. Frame 1 with distress: Instant={f1.instant_level}, Smoothed={f1.smoothed_level} (Expected: NORMAL)")
    assert f1.smoothed_level == "NORMAL"

    # Frame 2: second consecutive frame upgrades condition
    f2 = engine.evaluate_road_quality(
        potholes=distress_potholes, cracks=[], road_plane_meta=nominal_plane_meta,
        depth_m=planar_depth, calib=calib, observable_road_area_m2=40.0, stereo_quality=stereo_qual_good
    )
    print(f"3b. Frame 2 with distress: Instant={f2.instant_level}, Smoothed={f2.smoothed_level} (Expected: HIGH or ALARMING)")
    assert f2.smoothed_level in ["HIGH", "ALARMING"]

    # Test Case 4: Downward Hysteresis (Requires 4 consecutive clean frames to return to NORMAL)
    print("4. Testing Downward Hysteresis (4 frames down):")
    down_levels = []
    for i in range(1, 5):
        f_down = engine.evaluate_road_quality(
            potholes=[], cracks=[], road_plane_meta=nominal_plane_meta,
            depth_m=planar_depth, calib=calib, observable_road_area_m2=45.0, stereo_quality=stereo_qual_good
        )
        down_levels.append(f_down.smoothed_level)
        print(f"   Clean Frame {i}: Smoothed={f_down.smoothed_level}")
    
    # By frame 4, it must return to NORMAL
    assert down_levels[-1] == "NORMAL", f"Expected NORMAL by frame 4, got {down_levels[-1]}"
    print(">> Temporal Hysteresis: PASSED")

    # Diagnostic JSON export
    diag_record = {
        "engine_version": "BusSense Evidence-Based Road Quality v1.2",
        "operational_thresholds": BUSSENSE_OPERATIONAL_THRESHOLDS,
        "hysteresis_spec": {
            "upward_frames": 2,
            "downward_frames": 4,
            "instant_alert_depth_cm": 8.0
        },
        "test_results": {
            "clean_road_category": rq_clean.condition_category,
            "insufficient_data_status": rq_invalid.status,
            "upward_hysteresis_sequence": [f1.smoothed_level, f2.smoothed_level],
            "downward_hysteresis_sequence": down_levels
        }
    }

    diag_file = eval_dir / "road_condition_diagnostic.json"
    with open(diag_file, "w") as f:
        json.dump(diag_record, f, indent=2)
    print(f"\nSaved road condition diagnostic JSON to: {diag_file}")

if __name__ == "__main__":
    audit_road_condition()
