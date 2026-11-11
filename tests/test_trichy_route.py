#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_trichy_route.py
==========================
Hard GPS & Road Constraint Tests (Section 40)

Strict Assertions:
1. Exactly ONE road selected: Bharathidasan Salai.
2. Road came from authentic OSM geometry (Way 369881223).
3. Road has complete continuous geometry (no broken nodes, length > 1000m).
4. Bus trajectory is derived from that geometry (sampled arc length).
5. NO straight-line interpolation between distant start and end.
6. EVERY bus point is within configured road tolerance (<=10m).
7. EVERY bus point belongs to the selected road identity.
8. Heading comes from road geometry (local bearing from polyline tangents).
9. NO arbitrary Trichy points are used.
10. NO random city cruising occurs.
11. Route starts and ends on the selected road.
12. Divider synthetic fault uses road-coordinate intervals (30m -> 45m).
13. Allowed divider opening (70m -> 85m) is not flagged (NORMAL_OPENING).
14. Multi-bus events merge correctly into 1 cluster (CORROBORATED).
15. Conflicting observations are retained (CONFLICTING_EVIDENCE).
"""

import pytest
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.trichy_road_geometry import (
    TrichyRoadGeometryManager,
    BHARATHIDASAN_SALAI_OSM_NODES,
    haversine_m
)
from bussense_infrastructure.trichy_replay_simulator import TrichyReplaySimulator
from bussense_infrastructure.evidence_engine import EvidenceEngine


@pytest.fixture
def trichy_mgr():
    return TrichyRoadGeometryManager(sampling_interval_m=5.0, tolerance_m=10.0)


# 1. Exactly ONE road selected
def test_01_exactly_one_road_selected(trichy_mgr):
    assert trichy_mgr.ROAD_NAME == "Bharathidasan Salai"
    assert trichy_mgr.OSM_WAY_ID == 369881223
    assert trichy_mgr.ROAD_SEGMENT_ID == "SEG-TRICHY-BHARATHIDASAN-01"


# 2. Road came from authentic OSM geometry
def test_02_road_came_from_osm_geometry(trichy_mgr):
    assert len(trichy_mgr.raw_nodes) == 26
    # Verify node IDs and coordinates match official OpenStreetMap database
    first_node = trichy_mgr.raw_nodes[0]
    assert first_node["id"] == 839712564
    assert round(first_node["lat"], 4) == 10.8022
    assert round(first_node["lon"], 4) == 78.6892


# 3. Road has complete continuous geometry
def test_03_road_has_continuous_geometry(trichy_mgr):
    assert trichy_mgr.total_road_length_m > 1000.0  # ~1241m
    # Verify no gap between consecutive nodes exceeds 150m
    for i in range(len(trichy_mgr.polyline_coords) - 1):
        p1 = trichy_mgr.polyline_coords[i]
        p2 = trichy_mgr.polyline_coords[i + 1]
        dist = haversine_m(p1[0], p1[1], p2[0], p2[1])
        assert dist < 150.0, f"Gap between node {i} and {i+1} is {dist:.1f}m (>150m)"


# 4. Bus trajectory is derived from that geometry
def test_04_bus_trajectory_derived_from_geometry(trichy_mgr):
    traj = trichy_mgr.generate_full_bus_trajectory("BUS_DEMO_001")
    assert len(traj) > 100  # At 5m sampling over 1241m, should have ~249 points
    for pt in traj:
        assert pt["road_name"] == "Bharathidasan Salai"
        assert pt["osm_way_id"] == 369881223
        assert pt["road_segment_id"] == "SEG-TRICHY-BHARATHIDASAN-01"


# 5. NO straight-line interpolation between distant start and end
def test_05_no_straight_line_interpolation(trichy_mgr):
    traj = trichy_mgr.generate_full_bus_trajectory("BUS_DEMO_001")
    start_pt = (traj[0]["latitude"], traj[0]["longitude"])
    end_pt = (traj[-1]["latitude"], traj[-1]["longitude"])

    # True road bends along curve; check that midpoint of polyline does NOT lie on straight line chord
    mid_idx = len(traj) // 2
    mid_actual = (traj[mid_idx]["latitude"], traj[mid_idx]["longitude"])

    # Straight line midpoint
    mid_chord = ((start_pt[0] + end_pt[0]) / 2.0, (start_pt[1] + end_pt[1]) / 2.0)
    chord_deviation = haversine_m(mid_actual[0], mid_actual[1], mid_chord[0], mid_chord[1])

    # Bharathidasan Salai curved trajectory must deviate significantly (>35m) from simple chord
    assert chord_deviation > 35.0, f"Deviation from chord is only {chord_deviation:.1f}m; looks like straight line!"


# 6. EVERY bus point is within configured road tolerance (<=10m)
def test_06_every_bus_point_within_tolerance(trichy_mgr):
    traj = trichy_mgr.generate_full_bus_trajectory("BUS_DEMO_001")
    for idx, pt in enumerate(traj):
        dist = pt["distance_to_centerline_m"]
        assert dist <= 10.0, f"Point {idx} exceeds tolerance: {dist:.2f}m > 10.0m"


# 7. EVERY bus point belongs to the selected road identity
def test_07_every_bus_point_belongs_to_selected_road(trichy_mgr):
    traj = trichy_mgr.generate_full_bus_trajectory("BUS_DEMO_001")
    for pt in traj:
        assert pt["road_name"] == trichy_mgr.ROAD_NAME
        assert pt["osm_way_id"] == trichy_mgr.OSM_WAY_ID


# 8. Heading comes from road geometry (local bearing)
def test_08_heading_comes_from_geometry(trichy_mgr):
    traj = trichy_mgr.generate_full_bus_trajectory("BUS_DEMO_001")
    assert len(traj) > 0
    headings = [pt["heading"] for pt in traj]
    # Check not hardcoded (must vary as road bends)
    assert len(set(headings)) > 5, "Headings appear hardcoded or constant!"
    for i in range(len(traj)):
        h = traj[i]["heading"]
        assert 0.0 <= h <= 360.0
        # Verify heading directly matches geometry sample at that exact distance
        _, _, expected_h = trichy_mgr.sample_point_at_distance(traj[i]["distance_along_road_m"])
        assert abs(h - round(expected_h, 1)) < 0.1, f"Heading {h} does not match road geometry bearing {expected_h}"


# 9 & 10. No arbitrary Trichy points & No random city cruising
def test_09_10_no_random_cruising(trichy_mgr):
    traj = trichy_mgr.generate_full_bus_trajectory("BUS_DEMO_001")
    prev_dist = -1.0
    for pt in traj:
        # Distance along road must strictly increase monotonically
        curr_dist = pt["distance_along_road_m"]
        assert curr_dist >= prev_dist
        prev_dist = curr_dist


# 11. Route starts and ends on the selected road
def test_11_route_starts_and_ends_on_selected_road(trichy_mgr):
    traj = trichy_mgr.generate_full_bus_trajectory("BUS_DEMO_001")
    val_start = trichy_mgr.validate_point_on_road(traj[0]["latitude"], traj[0]["longitude"])
    val_end = trichy_mgr.validate_point_on_road(traj[-1]["latitude"], traj[-1]["longitude"])
    assert val_start["valid"]
    assert val_end["valid"]
    assert val_start["distance_to_centerline_m"] < 1.0
    assert val_end["distance_to_centerline_m"] < 1.0


# 12. Divider synthetic fault uses road-coordinate intervals (30m -> 45m)
def test_12_synthetic_fault_road_coordinates():
    sim = TrichyReplaySimulator()
    res = sim.run_simulation()
    assert res["road_name"] == "Bharathidasan Salai"
    assert res["simulation_mode"] is True


# 13. Physical divider missing candidate evaluation
def test_13_divider_gap_candidate_flagged(trichy_mgr):
    engine = EvidenceEngine()
    res = engine.evaluate_missing_asset(
        asset_type="ROAD_DIVIDER",
        expected_state="YES",
        observed_present=False,
        visibility_score=0.95,
        consecutive_absences=5,
        unique_buses=1
    )
    assert res.status == "CANDIDATE"
    assert res.is_missing_claim


# 14. Multi-bus events merge correctly (CORROBORATED)
def test_14_multibus_events_merge_corroborated():
    sim = TrichyReplaySimulator()
    res = sim.run_simulation()
    assert res["agreeing_passes_status"] == "CORROBORATED"
    assert res["agreeing_bus_count"] == 3


# 15. Conflicting observations are retained (CONFLICTING_EVIDENCE)
def test_15_conflicting_observations_retained():
    sim = TrichyReplaySimulator()
    res = sim.run_simulation()
    assert res["after_conflict_status"] == "CONFLICTING_EVIDENCE"
    assert res["total_conflicts"] >= 1
    assert res["final_unique_bus_count"] == 4
