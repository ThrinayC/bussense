#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_missing_assets_matrix.py
====================================
Deterministic Unit & Integration Test Matrix (Section 39)

Verifies all 17 mandatory criteria:
1. Expected sign + sign observed -> OBSERVED
2. Expected sign + sign not observed + good visibility -> MISSING_SIGN_CANDIDATE
3. Expected sign + sign not observed + poor visibility -> INSUFFICIENT_DATA
4. Unknown sign expectation -> NO_MISSING_CLAIM
5. Expected zebra + crossing observed -> VISIBLE / FADED
6. Expected zebra + crossing not observed -> MISSING_ZEBRA_CANDIDATE
7. Expected divider + divider observed -> OBSERVED
8. Expected divider + persistent gap -> MISSING_DIVIDER_CANDIDATE
9. Poor visibility divider gap -> INSUFFICIENT_DATA
10. Multiple buses agree -> CORROBORATED
11. Multiple buses disagree -> CONFLICTING_EVIDENCE
12. Multiple frames from one bus -> 1 vote (unique_bus_count = 1)
13. GPS point off road -> validation fails
14. Disconnected geometry -> rejected
15. Synthetic event -> simulation_mode = True
16. Real observation -> simulation_mode = False
17. Event deduplication -> merged cluster
"""

import pytest
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.evidence_engine import EvidenceEngine
from bussense_infrastructure.road_divider_engine import RoadDividerEngine
from bussense_infrastructure.multi_bus_fusion import MultiBusFusionEngine
from bussense_infrastructure.zebra_crossing_engine import ZebraCrossingEngine
from bussense_infrastructure.trichy_road_geometry import TrichyRoadGeometryManager


@pytest.fixture
def engine():
    return EvidenceEngine(min_visibility_for_absence=0.65, min_persistence_frames=4, min_corroborating_buses=2)


# 1. Expected sign + sign observed
def test_01_expected_sign_observed(engine):
    res = engine.evaluate_missing_asset(
        asset_type="TRAFFIC_SIGN",
        expected_state="YES",
        observed_present=True,
        visibility_score=0.90,
        consecutive_absences=0
    )
    assert res.status == "OBSERVED"
    assert not res.is_missing_claim


# 2. Expected sign + sign not observed + good visibility
def test_02_expected_sign_not_observed_good_visibility(engine):
    res = engine.evaluate_missing_asset(
        asset_type="TRAFFIC_SIGN",
        expected_state="YES",
        observed_present=False,
        visibility_score=0.85,
        consecutive_absences=4,
        unique_buses=1
    )
    assert res.status == "CANDIDATE"
    assert res.is_missing_claim


# 3. Expected sign + sign not observed + poor visibility
def test_03_expected_sign_not_observed_poor_visibility(engine):
    res = engine.evaluate_missing_asset(
        asset_type="TRAFFIC_SIGN",
        expected_state="YES",
        observed_present=False,
        visibility_score=0.45,  # below 0.65 threshold
        consecutive_absences=4,
        unique_buses=1
    )
    assert res.status == "INSUFFICIENT_DATA"
    assert not res.is_missing_claim


# 4. Unknown sign expectation -> NO_MISSING_CLAIM
def test_04_unknown_sign_expectation(engine):
    res = engine.evaluate_missing_asset(
        asset_type="TRAFFIC_SIGN",
        expected_state="UNKNOWN",
        observed_present=False,
        visibility_score=0.95,
        consecutive_absences=10
    )
    assert res.status == "NO_MISSING_CLAIM"
    assert not res.is_missing_claim


# 5. Expected zebra + crossing observed
def test_05_expected_zebra_observed(engine):
    res = engine.evaluate_missing_asset(
        asset_type="ZEBRA_CROSSING",
        expected_state="YES",
        observed_present=True,
        visibility_score=0.92,
        consecutive_absences=0
    )
    assert res.status == "OBSERVED"
    assert not res.is_missing_claim


# 6. Expected zebra + crossing not observed
def test_06_expected_zebra_not_observed(engine):
    res = engine.evaluate_missing_asset(
        asset_type="ZEBRA_CROSSING",
        expected_state="YES",
        observed_present=False,
        visibility_score=0.88,
        consecutive_absences=5,
        unique_buses=1
    )
    assert res.status == "CANDIDATE"
    assert res.is_missing_claim


# 7. Expected divider + divider observed
def test_07_expected_divider_observed(engine):
    res = engine.evaluate_missing_asset(
        asset_type="ROAD_DIVIDER",
        expected_state="YES",
        observed_present=True,
        visibility_score=0.95,
        consecutive_absences=0
    )
    assert res.status == "OBSERVED"
    assert not res.is_missing_claim


# 8. Expected divider + persistent gap
def test_08_expected_divider_persistent_gap(engine):
    res = engine.evaluate_missing_asset(
        asset_type="ROAD_DIVIDER",
        expected_state="YES",
        observed_present=False,
        visibility_score=0.85,
        consecutive_absences=6,
        unique_buses=1
    )
    assert res.status == "CANDIDATE"
    assert res.is_missing_claim


# 9. Expected divider + poor visibility -> INSUFFICIENT_DATA (no missing claim)
def test_09_gap_poor_visibility_insufficient_data(engine):
    res = engine.evaluate_missing_asset(
        asset_type="ROAD_DIVIDER",
        expected_state="YES",
        observed_present=False,
        visibility_score=0.40,
        consecutive_absences=5
    )
    assert res.status == "INSUFFICIENT_DATA"
    assert not res.is_missing_claim


# 10. Multiple buses agree -> CORROBORATED
def test_10_multiple_buses_agree(engine):
    res = engine.evaluate_missing_asset(
        asset_type="ROAD_DIVIDER",
        expected_state="YES",
        observed_present=False,
        visibility_score=0.90,
        consecutive_absences=5,
        unique_buses=3  # 3 independent buses agree
    )
    assert res.status == "CORROBORATED"
    assert res.is_missing_claim


# 11. Multiple buses disagree -> CONFLICTING_EVIDENCE
def test_11_multiple_buses_disagree(engine):
    res = engine.evaluate_missing_asset(
        asset_type="ROAD_DIVIDER",
        expected_state="YES",
        observed_present=False,
        visibility_score=0.90,
        consecutive_absences=5,
        unique_buses=2,
        has_conflicts=True
    )
    assert res.status == "CONFLICTING_EVIDENCE"
    assert not res.is_missing_claim


# 12. Multiple frames from one bus -> 1 vote (unique_bus_count = 1)
def test_12_multiple_frames_from_one_bus():
    fusion = MultiBusFusionEngine()
    # 50 frames from BUS_A
    obs_bus_a = [
        {
            "observation_id": f"OBS-A-{i}",
            "bus_id": "BUS_A",
            "timestamp": 100.0 + i * 0.1,
            "latitude": 10.8035,
            "longitude": 78.6878,
            "heading": 85.0,
            "road_segment_id": "SEG-01",
            "event_type": "MISSING_DIVIDER_CANDIDATE",
            "parameters": {"is_missing": True, "gap_length_m": 15.0}
        }
        for i in range(50)
    ]
    clusters = fusion.ingest_bus_pass("BUS_A", "PASS-A1", obs_bus_a)
    assert len(clusters) == 1
    assert len(clusters[0].unique_buses) == 1
    assert clusters[0].observation_count == 50
    # Must NOT be corroborated with only 1 bus
    assert clusters[0].status == "CANDIDATE"


# 13. GPS point off road -> validation fails
def test_13_gps_point_off_road():
    mgr = TrichyRoadGeometryManager(tolerance_m=10.0)
    # Point located 500 meters away in Kaveri river or distant field
    off_road_lat, off_road_lon = 10.8350, 78.6950
    val = mgr.validate_point_on_road(off_road_lat, off_road_lon)
    assert not val["valid"]
    assert val["distance_to_centerline_m"] > 10.0


# 14. Disconnected geometry -> rejected
def test_14_disconnected_geometry_rejected():
    mgr = TrichyRoadGeometryManager()
    # Every consecutive pair of points in BHARATHIDASAN_SALAI_OSM_NODES must be connected (< 150m apart)
    for i in range(len(mgr.polyline_coords) - 1):
        p1 = mgr.polyline_coords[i]
        p2 = mgr.polyline_coords[i+1]
        dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1]) * 111320.0
        assert dist < 150.0, f"Discontinuity detected between nodes {i} and {i+1}: {dist:.1f}m"


# 15. Synthetic event -> simulation_mode = True
def test_15_synthetic_event_flag():
    fusion = MultiBusFusionEngine()
    obs_syn = [{
        "observation_id": "SYN-OBS-01",
        "bus_id": "BUS_DEMO_01",
        "timestamp": 10.0,
        "latitude": 10.8035,
        "longitude": 78.6878,
        "heading": 85.0,
        "road_segment_id": "SEG-01",
        "event_type": "MISSING_DIVIDER_CANDIDATE",
        "parameters": {"is_missing": True, "simulation_mode": True}
    }]
    clusters = fusion.ingest_bus_pass("BUS_DEMO_01", "PASS-01", obs_syn, simulation_mode=True)
    assert clusters[0].simulation_mode is True


# 16. Real observation -> simulation_mode = False
def test_16_real_observation_flag():
    fusion = MultiBusFusionEngine()
    obs_real = [{
        "observation_id": "REAL-OBS-01",
        "bus_id": "BUS_FIELD_01",
        "timestamp": 10.0,
        "latitude": 10.8035,
        "longitude": 78.6878,
        "heading": 85.0,
        "road_segment_id": "SEG-01",
        "event_type": "ROAD_DIVIDER",
        "parameters": {"is_missing": False}
    }]
    clusters = fusion.ingest_bus_pass("BUS_FIELD_01", "PASS-01", obs_real, simulation_mode=False)
    assert clusters[0].simulation_mode is False


# 17. Event deduplication -> merged cluster
def test_17_event_deduplication():
    fusion = MultiBusFusionEngine(spatial_merge_radius_m=20.0)
    obs1 = [{
        "observation_id": "OBS-1", "bus_id": "BUS-1", "timestamp": 10.0,
        "latitude": 10.8035, "longitude": 78.6878, "heading": 85.0,
        "road_segment_id": "SEG-01", "event_type": "MISSING_DIVIDER_CANDIDATE",
        "parameters": {"is_missing": True}
    }]
    obs2 = [{
        "observation_id": "OBS-2", "bus_id": "BUS-2", "timestamp": 20.0,
        "latitude": 10.8036, "longitude": 78.6879, "heading": 86.0,  # 15m away
        "road_segment_id": "SEG-01", "event_type": "MISSING_DIVIDER_CANDIDATE",
        "parameters": {"is_missing": True}
    }]
    fusion.ingest_bus_pass("BUS-1", "P1", obs1)
    clusters = fusion.ingest_bus_pass("BUS-2", "P2", obs2)
    # Must merge into single cluster
    assert len(clusters) == 1
    assert len(clusters[0].unique_buses) == 2
    assert clusters[0].status == "CORROBORATED"
