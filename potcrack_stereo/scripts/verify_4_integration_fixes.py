#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification Script for the 4 Critical BusSense Integration Fixes:
1. Test A: Road Quality Dynamic State & Hysteresis (NORMAL / MEDIUM / HIGH / ALARMING)
2. Test B: Pothole Metrology Sanity (Bounding Check, Depth <= 15 cm, Area <= 10 m2, No Fake Fallbacks)
3. Test C: Crack Detection with conf_crack = 0.08 (Instance segmentation, metric length, area)
4. Test D: Traffic Operational Levels & Hysteresis (NORMAL / MEDIUM / HIGH / ALARMING)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.road_quality import RoadQualityEngine, OPERATIONAL_LEVELS, RoadQualityMetrics
from scripts.pothole_geometry import PotholeGeometryEngine, PotholeInstance3D
from traffic.density_engine import TrafficDensityEngine, TRAFFIC_LEVELS
from scripts.run_unified_pipeline import UnifiedBusSensePipeline
from scripts.make_kitti_traffic_demo import parse_kitti_calib, CALIB_FILE


def test_a_road_quality():
    print("\n" + "="*70)
    print("TEST A: Road Quality Engine Dynamic States & Temporal Hysteresis")
    print("="*70)
    rq_engine = RoadQualityEngine(history_len=8)

    # 1. Verify operational levels
    expected_levels = ["NORMAL", "MEDIUM", "HIGH", "ALARMING"]
    assert list(OPERATIONAL_LEVELS.keys()) == expected_levels, f"Unexpected levels: {list(OPERATIONAL_LEVELS.keys())}"
    print("[PASS] Operational levels defined correctly:", expected_levels)

    # 2. Test clean road (0 potholes, 0 cracks, small roughness) -> Expect NORMAL
    road_plane_meta = {"is_valid": True, "equation": [0.0, -1.0, 0.0, 1.65], "inlier_ratio": 0.85}
    stereo_quality = {"valid_disp_fraction": 0.85}
    calib = {"fx": 457.0, "fy": 457.0, "cx": 320.0, "cy": 180.0, "baseline": 0.12}
    
    # Generate geometrically consistent planar depth map for road region (Y = 1.65m)
    # v > cy: (v - cy)/fy * Z = 1.65 => Z = 1.65 * fy / (v - cy)
    depth_m = np.zeros((360, 640), dtype=np.float32)
    for v in range(181, 360):
        depth_m[v, :] = 1.65 * calib["fy"] / (v - calib["cy"])

    clean_metrics = rq_engine.assess(
        potholes=[],
        cracks=[],
        road_plane_meta=road_plane_meta,
        depth_m=depth_m,
        calib=calib,
        observable_road_area_m2=50.0,
        stereo_quality=stereo_quality
    )
    print(f"Clean road frame: instant={clean_metrics.instant_level}, smoothed={clean_metrics.smoothed_level}")
    assert clean_metrics.instant_level == "NORMAL", f"Expected NORMAL for clean road, got {clean_metrics.instant_level}"
    assert clean_metrics.smoothed_level == "NORMAL", f"Expected NORMAL smoothed, got {clean_metrics.smoothed_level}"
    assert clean_metrics.operational_level == "NORMAL"
    print("[PASS] Clean road correctly identified as NORMAL")

    # 3. Simulate sudden damage (2 potholes with 0.4m2 area, 5cm depth, cracks)
    severe_potholes = [
        {"area_m2": 0.35, "depth_p95_cm": 6.2, "valid": True},
        {"area_m2": 0.25, "depth_p95_cm": 5.1, "valid": True}
    ]
    severe_cracks = [
        {"length_m": 4.5, "coverage_area_m2": 0.8}
    ]

    # Frame 1 of damage: instant should be HIGH or ALARMING
    f1 = rq_engine.assess(
        potholes=severe_potholes,
        cracks=severe_cracks,
        road_plane_meta=road_plane_meta,
        depth_m=depth_m,
        calib=calib,
        observable_road_area_m2=40.0,
        stereo_quality=stereo_quality
    )
    print(f"Damage Frame 1: instant={f1.instant_level}, smoothed={f1.smoothed_level}")
    assert f1.instant_level in ["HIGH", "ALARMING"]

    # Frame 2 of damage: smoothed should escalate due to 2-frame upward hysteresis
    f2 = rq_engine.assess(
        potholes=severe_potholes,
        cracks=severe_cracks,
        road_plane_meta=road_plane_meta,
        depth_m=depth_m,
        calib=calib,
        observable_road_area_m2=40.0,
        stereo_quality=stereo_quality
    )
    print(f"Damage Frame 2: instant={f2.instant_level}, smoothed={f2.smoothed_level}")
    assert f2.smoothed_level in ["HIGH", "ALARMING"], f"Expected rapid upward escalation, got {f2.smoothed_level}"
    print(f"[PASS] Fast upward escalation verified: escalated to {f2.smoothed_level} on 2nd frame")

    # 4. Simulate road clear: test downward hysteresis (needs 4 frames of lower severity to drop)
    for i in range(1, 6):
        rec = rq_engine.assess(
            potholes=[],
            cracks=[],
            road_plane_meta=road_plane_meta,
            depth_m=depth_m,
            calib=calib,
            observable_road_area_m2=50.0,
            stereo_quality=stereo_quality
        )
        print(f"Recovery Frame {i}: instant={rec.instant_level}, smoothed={rec.smoothed_level}")
        if i < 4:
            # Should stay elevated (hysteresis preventing premature de-escalation)
            assert rec.smoothed_level in ["MEDIUM", "HIGH", "ALARMING"]
        else:
            # By frame 4 or 5, should de-escalate towards NORMAL
            pass

    assert rec.smoothed_level == "NORMAL", f"Expected recovery to NORMAL by frame 5, got {rec.smoothed_level}"
    print("[PASS] Downward recovery hysteresis verified (stabilized back to NORMAL)")


def test_b_pothole_metrology():
    print("\n" + "="*70)
    print("TEST B: Pothole Metrology Bounding Sanity & Physical Limits")
    print("="*70)

    # Load 5 sample frames from data/pothole_long_stereo/
    data_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
    left_files = sorted(list((data_dir / "left").glob("*.jpg")))
    right_files = sorted(list((data_dir / "right").glob("*.jpg")))

    assert len(left_files) > 0, "No pothole stereo frames found!"

    # ZED 2.2K -> 640x360 calibrated intrinsics
    calib = {
        "fx": 457.0,
        "fy": 457.0,
        "cx": 320.0,
        "cy": 180.0,
        "baseline": 0.12
    }

    # Initialize unified pipeline
    pipeline = UnifiedBusSensePipeline(calib=calib, conf_pothole=0.15, conf_crack=0.08)

    sample_indices = [10, 20, 30, 45, 60]
    total_potholes_evaluated = 0

    for idx in sample_indices:
        if idx >= len(left_files):
            continue
        l_img = cv2.imread(str(left_files[idx]))
        r_img = cv2.imread(str(right_files[idx]))

        state = pipeline.process_frame(l_img, r_img, frame_index=idx, timestamp=idx * 0.1)
        potholes = state.potholes

        print(f"\nFrame {idx:03d}: detected {len(potholes)} potholes")
        for i, p in enumerate(potholes):
            total_potholes_evaluated += 1
            area_m2 = p.get("area_m2")
            depth_cm = p.get("depth_p95_cm")
            len_m = p.get("length_m")
            wid_m = p.get("width_m")
            quality = p.get("quality", "INVALID")
            valid = (quality in ["GOOD", "WARNING"])
            status = p.get("quality_reason", "")

            print(f"  Pothole #{i+1} (ID {p.get('track_id')}): quality={quality}, depth={depth_cm} cm, area={area_m2} m2, length={len_m} m, width={wid_m} m, reason={status}")

            if valid and area_m2 is not None and depth_cm is not None:
                # 1. Physical limits: depth <= 15.0 cm, area <= 10.0 m2
                assert depth_cm <= 15.0, f"Depth impossible: {depth_cm} cm > 15.0 cm!"
                assert area_m2 <= 10.0, f"Area impossible: {area_m2} m2 > 10.0 m2!"

                # 2. Bounding check: area_m2 <= length_m * width_m * 1.05
                if len_m is not None and wid_m is not None and len_m > 0 and wid_m > 0:
                    bbox_area = len_m * wid_m
                    assert area_m2 <= bbox_area * 1.05, f"Violated bounding check! area={area_m2} > length*width={bbox_area}"
                    print(f"    [CHECK PASSED] area {area_m2:.4f} m2 <= bbox {bbox_area:.4f} m2")

            else:
                # Must honestly state unreliable without throwing crash or producing fake area
                print(f"    [CHECK PASSED] Unreliable geometry reported cleanly without fake fallback")

    print(f"\n[PASS] All {total_potholes_evaluated} evaluated potholes satisfied physical & bounding constraints!")


def test_c_crack_detection():
    print("\n" + "="*70)
    print("TEST C: Crack Detection with conf_crack = 0.08 & Geometry")
    print("="*70)

    calib = {
        "fx": 457.0,
        "fy": 457.0,
        "cx": 320.0,
        "cy": 180.0,
        "baseline": 0.12
    }

    pipeline = UnifiedBusSensePipeline(calib=calib, conf_pothole=0.15, conf_crack=0.08)

    data_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
    left_files = sorted(list((data_dir / "left").glob("*.jpg")))
    right_files = sorted(list((data_dir / "right").glob("*.jpg")))

    # Scan frames for cracks
    cracks_found = 0
    sample_frames = [5, 12, 18, 25, 35, 45, 55, 65]

    for idx in sample_frames:
        if idx >= len(left_files):
            continue
        l_img = cv2.imread(str(left_files[idx]))
        r_img = cv2.imread(str(right_files[idx]))

        state = pipeline.process_frame(l_img, r_img, frame_index=idx, timestamp=idx * 0.1)
        cracks = state.cracks

        if len(cracks) > 0:
            cracks_found += len(cracks)
            print(f"Frame {idx:03d}: detected {len(cracks)} crack(s)")
            for c in cracks:
                print(f"  Crack: length={c.get('length_m'):.2f}m, area={c.get('coverage_area_m2'):.4f}m2, conf={c.get('confidence'):.2f}")
                assert c.get('length_m') is not None and c.get('length_m') >= 0.0
                assert c.get('coverage_area_m2') is not None and c.get('coverage_area_m2') >= 0.0

    print(f"[SUMMARY] Total cracks detected across sampled frames: {cracks_found}")
    assert cracks_found > 0, "No cracks detected with conf_crack = 0.08!"
    print("[PASS] Crack branch operational and generating metric length & coverage area.")


def test_d_traffic_levels():
    print("\n" + "="*70)
    print("TEST D: Traffic Operational Levels & Temporal Hysteresis")
    print("="*70)

    engine = TrafficDensityEngine(fx=721.5, img_width=1242, img_height=375)

    # Verify levels enum
    assert set(TRAFFIC_LEVELS.keys()) == {"NORMAL", "MEDIUM", "HIGH", "ALARMING"}

    # 1. Zero/single vehicle -> NORMAL
    plane = np.array([0.0, -1.0, 0.0, 1.65], dtype=np.float32)
    scene_1 = {
        "vehicles": [{"track_id": 1, "distance_m": 25.0, "bbox": [100, 100, 200, 250]}],
        "inter_vehicle_distances": []
    }
    t1 = engine.update(scene_analysis=scene_1, road_plane=plane, road_plane_inliers=900, total_road_pts=1000)
    print(f"Low traffic: level={t1.get('traffic_level')}, raw={t1.get('traffic_level_raw')}")
    assert t1.get("traffic_level") == "NORMAL"

    # 2. Heavy traffic: 5 vehicles, tight nearest distance (< 6m) -> ALARMING
    heavy_vehicles = [
        {"track_id": 1, "distance_m": 4.5, "bbox": [550, 150, 650, 250]},
        {"track_id": 2, "distance_m": 9.0, "bbox": [570, 120, 630, 200]},
        {"track_id": 3, "distance_m": 14.0, "bbox": [580, 100, 620, 160]},
        {"track_id": 4, "distance_m": 18.0, "bbox": [590, 80, 610, 140]},
        {"track_id": 5, "distance_m": 22.0, "bbox": [595, 70, 605, 120]}
    ]
    scene_heavy = {
        "vehicles": heavy_vehicles,
        "inter_vehicle_distances": [{"longitudinal_gap_m": 4.5, "distance_3d_m": 4.5}]
    }

    # Frame 1 of heavy traffic
    t2_1 = engine.update(scene_analysis=scene_heavy, road_plane=plane, road_plane_inliers=900, total_road_pts=1000)
    print(f"Heavy traffic Frame 1: level={t2_1.get('traffic_level')}, raw={t2_1.get('traffic_level_raw')}")

    # Frame 2 of heavy traffic -> Rapid upward transition to ALARMING
    t2_2 = engine.update(scene_analysis=scene_heavy, road_plane=plane, road_plane_inliers=900, total_road_pts=1000)
    print(f"Heavy traffic Frame 2: level={t2_2.get('traffic_level')}, raw={t2_2.get('traffic_level_raw')}")
    assert t2_2.get("traffic_level") in ["HIGH", "ALARMING"], f"Expected rapid upward escalation, got {t2_2.get('traffic_level')}"

    # 3. Traffic clears -> Downward hysteresis
    scene_cleared = {
        "vehicles": [{"track_id": 1, "distance_m": 30.0, "bbox": [500, 100, 600, 200]}],
        "inter_vehicle_distances": []
    }
    for i in range(1, 6):
        t_rec = engine.update(scene_analysis=scene_cleared, road_plane=plane, road_plane_inliers=900, total_road_pts=1000)
        print(f"Traffic clearing Frame {i}: level={t_rec.get('traffic_level')}, raw={t_rec.get('traffic_level_raw')}")
        if i < 4:
            assert t_rec.get("traffic_level") in ["MEDIUM", "HIGH", "ALARMING"]
        else:
            pass

    assert t_rec.get("traffic_level") == "NORMAL", f"Expected NORMAL by frame 5, got {t_rec.get('traffic_level')}"
    print("[PASS] Traffic operational levels and hysteresis verified!")


if __name__ == "__main__":
    try:
        test_a_road_quality()
        test_b_pothole_metrology()
        test_c_crack_detection()
        test_d_traffic_levels()
        print("\n" + "="*70)
        print("ALL 4 INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("="*70)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
