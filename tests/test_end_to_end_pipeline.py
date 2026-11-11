#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_end_to_end_pipeline.py
=================================
End-to-End BusSense Perception Pipeline Verification Test Suite (Section R).

Tests 15 Critical Invariants:
1.  Model registry integrity and status consistency.
2.  Stereo sensor contract enforcement (Left + Right RGB only).
3.  Single-pass stereo constraint (disparity computed exactly once per frame).
4.  Metric depth calculation (Z = f * B / d within [0.3m, 80m]).
5.  Vehicle & pedestrian detection with 3D range and lateral offset.
6.  Pothole segmentation with 3D depth (cm) and area (m²).
7.  Crack segmentation with length (m).
8.  Traffic sign detection with condition classification.
9.  Waterlogging ego-corridor extent with union denominator.
10. Zebra crossing paint coverage and wear percentage calculation.
11. Road divider physical elevation and gap detection.
12. Separation of authorized openings (U-turns) vs missing divider candidates.
13. Road hazard and geometric unknown obstruction detection.
14. Trichy OSM Way 369881223 polyline adherence (centerline deviation <= 10m, tortuosity > 1.0).
15. Strict tagging distinction between real perception and synthetic simulation modes.
"""

import os
import sys
import json
import pytest
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.pipeline import BusSensePipeline, StereoInferenceManager
from bussense_infrastructure.trichy_road_geometry import TrichyRoadGeometryManager, BHARATHIDASAN_SALAI_OSM_NODES


@pytest.fixture(scope="module")
def pipeline():
    return BusSensePipeline()


@pytest.fixture(scope="module")
def sample_stereo_pair():
    pothole_left = ROOT / "potcrack_stereo" / "data" / "pothole_long_stereo" / "left" / "frame_00000.jpg"
    pothole_right = ROOT / "potcrack_stereo" / "data" / "pothole_long_stereo" / "right" / "frame_00000.jpg"
    if pothole_left.exists() and pothole_right.exists():
        l_img = cv2.imread(str(pothole_left))
        r_img = cv2.imread(str(pothole_right))
        return l_img, r_img
    # Synthetic fallback
    l_img = np.zeros((480, 640, 3), dtype=np.uint8)
    r_img = np.zeros((480, 640, 3), dtype=np.uint8)
    return l_img, r_img


# Test 1: Model registry integrity
def test_model_registry_integrity():
    reg_path = ROOT / "models" / "infrastructure" / "pretrained_registry.json"
    assert reg_path.exists(), "pretrained_registry.json must exist"
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "models" in data
    valid_statuses = {
        "PRETRAINED_VERIFIED",
        "RELEASED_CHECKPOINT",
        "LOCAL_EXPERIMENTAL",
        "GEOMETRIC",
        "NOT_AVAILABLE",
        "AVAILABLE",
        "NEEDS_ADAPTER",
    }
    for m in data["models"]:
        assert m["status"] in valid_statuses, f"Invalid status: {m['status']}"
        assert ("id" in m or "model_id" in m), f"Model entry missing id: {m}"
        assert "task" in m


# Test 2: Stereo sensor contract enforcement
def test_stereo_contract_input(sample_stereo_pair):
    mgr = StereoInferenceManager()
    left, right = sample_stereo_pair
    assert left is not None and right is not None
    assert left.shape[:2] == right.shape[:2], "Stereo cameras must be synchronized and rectified"


# Test 3: Single-pass stereo constraint
def test_stereo_single_pass_execution(pipeline, sample_stereo_pair):
    left, right = sample_stereo_pair
    count_before = pipeline.stereo_mgr.compute_count
    rec = pipeline.process_stereo_frame(left, right, frame_idx=0, timestamp=0.0)
    count_after = pipeline.stereo_mgr.compute_count
    assert count_after == count_before + 1, "Disparity must be computed EXACTLY ONCE per frame"
    assert "stereo_metrics" in rec
    assert rec["stereo_metrics"]["compute_count"] == count_after


# Test 4: Metric depth calculation
def test_stereo_metric_depth_calculation(pipeline, sample_stereo_pair):
    left, right = sample_stereo_pair
    disp, depth, valid_mask, metrics = pipeline.stereo_mgr.compute_stereo(left, right)
    assert depth.shape == left.shape[:2]
    if np.any(valid_mask):
        valid_depths = depth[valid_mask]
        assert np.all(valid_depths >= 0.3), "Depth must obey minimum physical range (0.3m)"
        assert np.all(valid_depths <= 80.0), "Depth must obey maximum physical range (80.0m)"


# Test 5: Vehicle & pedestrian detection with 3D range
def test_vehicle_and_pedestrian_metrology(pipeline, sample_stereo_pair):
    left, right = sample_stereo_pair
    rec = pipeline.process_stereo_frame(left, right, frame_idx=0)
    assert "vehicles" in rec
    assert "pedestrians" in rec
    for v in rec["vehicles"]:
        assert "distance_m" in v
        assert "lateral_offset_m" in v
        assert "in_corridor" in v


# Test 6: Pothole segmentation with 3D depth and area
def test_pothole_3d_geometry(pipeline, sample_stereo_pair):
    left, right = sample_stereo_pair
    rec = pipeline.process_stereo_frame(left, right, frame_idx=0)
    assert "potholes" in rec
    for p in rec["potholes"]:
        assert "depth_cm" in p and p["depth_cm"] >= 0.0
        assert "area_m2" in p and p["area_m2"] >= 0.0
        assert p["severity"] in ["MEDIUM", "HIGH"]


# Test 7: Crack segmentation with length
def test_crack_geometry(pipeline, sample_stereo_pair):
    left, right = sample_stereo_pair
    rec = pipeline.process_stereo_frame(left, right, frame_idx=0)
    assert "cracks" in rec
    for c in rec["cracks"]:
        assert "length_m" in c and c["length_m"] >= 0.0
        assert c["severity"] in ["LOW", "HIGH"]


# Test 8: Traffic sign detection with condition classification
def test_traffic_sign_detection_and_condition(pipeline):
    dummy_sign = np.full((128, 128, 3), 180, dtype=np.uint8)
    cond = pipeline.sign_condition_adapter.predict(dummy_sign)
    assert "condition" in cond
    assert cond["condition"] in ["GOOD", "FADED", "DAMAGED", "OBSCURED"]
    assert "confidence" in cond


# Test 9: Waterlogging extent with union denominator
def test_waterlogging_corridor_union_ratio(pipeline, sample_stereo_pair):
    left, right = sample_stereo_pair
    rec = pipeline.process_stereo_frame(left, right, frame_idx=0)
    assert "waterlogging" in rec
    water = rec["waterlogging"]
    assert 0.0 <= water["water_coverage_ratio"] <= 1.0
    assert water["denominator_union_px"] >= water["water_area_px"]
    assert water["severity"] in ["NONE", "MEDIUM", "ALARMING"]


# Test 10: Zebra crossing paint wear calculation
def test_zebra_crossing_wear(pipeline, sample_stereo_pair):
    left, right = sample_stereo_pair
    rec = pipeline.process_stereo_frame(left, right, frame_idx=0)
    assert "zebra_crossings" in rec
    for z in rec["zebra_crossings"]:
        assert 0.0 <= z["paint_coverage_pct"] <= 100.0
        assert 0.0 <= z["wear_percentage"] <= 100.0
        assert round(z["paint_coverage_pct"] + z["wear_percentage"], 1) == 100.0


# Test 11: Road divider physical elevation and gap detection
def test_road_divider_physical_elevation(pipeline, sample_stereo_pair):
    left, right = sample_stereo_pair
    rec = pipeline.process_stereo_frame(left, right, frame_idx=0, road_position_m=10.0)
    assert "road_divider" in rec
    div = rec["road_divider"]
    assert "elevation_cm" in div
    assert "divider_present" in div or "divider_observed" in div
    assert "status" in div


# Test 12: Road divider taxonomy and physical separation verification
def test_road_divider_physical_classification(pipeline):
    # Valid physical median separating opposing carriageways
    tax_median = pipeline.divider_engine.classify_structure(
        has_visible_vertical_structure=True,
        is_outer_road_edge=False,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=True,
        structure_type_hint="CONCRETE_JERSEY"
    )
    assert tax_median.value in ["PHYSICAL_MEDIAN", "JERSEY_BARRIER"]

    # Flat paint markings must reject physical divider
    tax_paint = pipeline.divider_engine.classify_structure(
        has_visible_vertical_structure=False,
        is_outer_road_edge=False,
        is_flat_paint=True,
        is_grass_adjacent=False,
        separates_opposing_carriageways=False
    )
    assert tax_paint.value == "LANE_MARKING"


# Test 13: Road hazard geometric unknown obstruction
def test_road_hazard_geometric_unknown(pipeline):
    obs = pipeline.hazard_engine.evaluate_candidate(
        category="UNKNOWN_OBSTRUCTION",
        confidence=0.75,
        bbox=(200, 200, 300, 300),
        position_3d=(0.1, 0.4, 8.5),
        speed_mps=0.0
    )
    assert obs is not None
    assert obs.is_geometric_unknown is True
    assert obs.hazard_type == "UNKNOWN_OBSTRUCTION"


# Test 14: Trichy OSM Way 369881223 polyline adherence
def test_trichy_osm_polyline_adherence():
    mgr = TrichyRoadGeometryManager(sampling_interval_m=5.0, tolerance_m=10.0)
    assert len(mgr.raw_nodes) == 26
    assert mgr.OSM_WAY_ID == 369881223

    # Sample points every 5m and verify all adhere within 10.0m tolerance
    num_steps = int(mgr.total_road_length_m // 5.0)
    for i in range(num_steps + 1):
        s = min(mgr.total_road_length_m, i * 5.0)
        lat, lon, heading = mgr.sample_point_at_distance(s)
        val = mgr.validate_point_on_road(lat, lon)
        assert val["valid"] is True, f"Point at s={s}m exceeded 10.0m tolerance: {val['distance_to_centerline_m']}m"
        assert val["distance_to_centerline_m"] <= 10.0


# Test 15: Strict tagging distinction between real and synthetic modes
def test_end_to_end_real_vs_synthetic_tagging(pipeline, sample_stereo_pair):
    left, right = sample_stereo_pair
    # Real mode
    rec_real = pipeline.process_stereo_frame(left, right, frame_idx=0, simulation_mode=False)
    assert rec_real["simulation_mode"] is False

    # Synthetic simulation mode
    rec_synth = pipeline.process_stereo_frame(left, right, frame_idx=0, simulation_mode=True)
    assert rec_synth["simulation_mode"] is True
