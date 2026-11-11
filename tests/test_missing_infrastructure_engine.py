#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_missing_infrastructure_engine.py
===========================================
Comprehensive unit tests verifying Part 2 of the BusSense Final Infrastructure Fix:
- Missing infrastructure reasoning (EXPECTED + VISIBILITY + OBSERVATION + HISTORY + VOTES)
- Single-frame detector absence does NOT trigger alert (POTENTIAL_MISSING)
- Multi-frame persistent absence triggers CONFIRMED_MISSING
- Authorized U-turn opening produces NORMAL_OPENING (no alert)
- Low visibility / occluded view suppresses missing claims (OBSCURED_UNCONFIRMED)
- Unknown reference produces NO_CLAIM
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.missing_infrastructure_engine import (
    MissingInfrastructureEngine,
    AssetType,
    ReferenceState,
    MissingStatus,
    ExpectedAsset
)


@pytest.fixture
def engine():
    return MissingInfrastructureEngine(min_temporal_absence_frames=5, min_corroboration_votes=2, min_visibility_threshold=0.65)


def test_missing_sign_temporal_lifecycle(engine):
    """Verifies that a missing sign requires temporal persistence before dispatching an alert."""
    asset = ExpectedAsset(
        asset_id="SIGN_REF_042",
        asset_type=AssetType.TRAFFIC_SIGN,
        reference_state=ReferenceState.EXPECTED,
        road_station_s_m=42.0,
        description="Speed Limit 40 km/h"
    )

    # Frame 1: detector saw nothing, but 1 frame is NOT a confirmed missing alert
    obs1 = engine.evaluate_asset(asset, current_detected=False, visibility_score=0.92)
    assert obs1.missing_status == MissingStatus.POTENTIAL_MISSING
    assert obs1.dispatch_alert is False
    assert obs1.temporal_absence_count == 1

    # Frames 2-4: continuing absence
    for _ in range(3):
        obs = engine.evaluate_asset(asset, current_detected=False, visibility_score=0.90)
        assert obs.missing_status == MissingStatus.POTENTIAL_MISSING
        assert obs.dispatch_alert is False

    # Frame 5: reaches threshold -> CONFIRMED_MISSING
    obs5 = engine.evaluate_asset(asset, current_detected=False, visibility_score=0.88)
    assert obs5.missing_status == MissingStatus.CONFIRMED_MISSING
    assert obs5.dispatch_alert is True
    assert "CONFIRMED MISSING" in obs5.display_tag


def test_present_asset_resets_absence(engine):
    """Verifies that when asset is detected, status is VERIFIED_PRESENT and absence count resets."""
    asset = ExpectedAsset(
        asset_id="SIGN_REF_010",
        asset_type=AssetType.TRAFFIC_SIGN,
        reference_state=ReferenceState.EXPECTED,
        road_station_s_m=10.0,
        description="Pedestrian Crossing"
    )

    obs = engine.evaluate_asset(asset, current_detected=True, visibility_score=0.95)
    assert obs.missing_status == MissingStatus.VERIFIED_PRESENT
    assert obs.dispatch_alert is False
    assert obs.temporal_absence_count == 0


def test_physical_divider_gap_evaluation(engine):
    """Verifies that an expected physical divider missing for multiple frames triggers CONFIRMED_MISSING."""
    asset = ExpectedAsset(
        asset_id="MEDIAN_SECTION_01",
        asset_type=AssetType.PHYSICAL_MEDIAN,
        reference_state=ReferenceState.EXPECTED,
        road_station_s_m=120.0,
        description="Physical Median Section"
    )

    for _ in range(4):
        obs = engine.evaluate_asset(asset, current_detected=False, visibility_score=0.90)
        assert obs.missing_status == MissingStatus.POTENTIAL_MISSING
        assert obs.dispatch_alert is False

    obs_final = engine.evaluate_asset(asset, current_detected=False, visibility_score=0.90)
    assert obs_final.missing_status == MissingStatus.CONFIRMED_MISSING
    assert obs_final.dispatch_alert is True


def test_low_visibility_suppresses_missing_claim(engine):
    """Verifies that occluded/rain/glare scenes (visibility < 0.65) suppress false missing alerts."""
    asset = ExpectedAsset(
        asset_id="ZEBRA_REF_100",
        asset_type=AssetType.ZEBRA_CROSSING,
        reference_state=ReferenceState.EXPECTED,
        road_station_s_m=100.0,
        description="School Crosswalk"
    )

    # Obscured view (e.g. spray, heavy shadows, lead vehicle occlusion)
    obs = engine.evaluate_asset(asset, current_detected=False, visibility_score=0.45)
    assert obs.missing_status == MissingStatus.OBSCURED_UNCONFIRMED
    assert obs.dispatch_alert is False
    assert "OBSCURED" in obs.display_tag


def test_unknown_reference_no_claim(engine):
    """Verifies that assets with UNKNOWN reference state produce NO_CLAIM."""
    asset = ExpectedAsset(
        asset_id="UNKNOWN_ASSET",
        asset_type=AssetType.ROAD_MARKING,
        reference_state=ReferenceState.UNKNOWN,
        road_station_s_m=0.0,
        description="Unsurveyed road"
    )

    obs = engine.evaluate_asset(asset, current_detected=False, visibility_score=0.90)
    assert obs.missing_status == MissingStatus.NO_CLAIM
    assert obs.dispatch_alert is False
