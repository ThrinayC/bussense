#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_road_marking_health_engine.py
========================================
Comprehensive unit test suite verifying Part 3, 4, 6 of the BusSense Final Infrastructure Fix:
- Strict 14 Test Cases per Part 6
- Invariant: lane marking != crosswalk
- Invariant: turn arrow != crosswalk
- Invariant: road text != crosswalk
- Invariant: bus-lane marking != crosswalk
- Invariant: occluded != faded (visibility-aware health)
- Segment-level health aggregation across 100m road section
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.road_marking_health_engine import (
    RoadMarkingHealthEngine,
    MarkingTaxonomy,
    MarkingCondition,
    RoadSegmentMarkingHealth
)


@pytest.fixture
def engine():
    return RoadMarkingHealthEngine(
        min_crosswalk_bars=4,
        good_paint_threshold=0.60,
        moderate_paint_threshold=0.35,
        faded_paint_threshold=0.15,
        contrast_threshold=0.30,
        occlusion_threshold=0.40
    )


# Case 1: Good lane lines
def test_case_01_good_lane_lines(engine):
    tax = engine.classify_marking_type(is_longitudinal_line=True, shape_type_hint="lane_line")
    assert tax == MarkingTaxonomy.LANE_LINE
    assert tax != MarkingTaxonomy.ZEBRA_CROSSING

    obs = engine.process_marking(
        marking_id="LANE_01", frame_idx=1, taxonomy=tax, bbox=(100, 200, 120, 500),
        paint_coverage_ratio=0.88, paint_contrast_ratio=0.75, continuity_ratio=0.95
    )
    assert obs.condition == MarkingCondition.GOOD


# Case 2: Worn lane line
def test_case_02_worn_lane_line(engine):
    tax = MarkingTaxonomy.LANE_LINE
    obs = engine.process_marking(
        marking_id="LANE_02", frame_idx=2, taxonomy=tax, bbox=(100, 200, 120, 500),
        paint_coverage_ratio=0.42, paint_contrast_ratio=0.50, continuity_ratio=0.70
    )
    assert obs.condition == MarkingCondition.ACCEPTABLE or obs.condition == MarkingCondition.MODERATE_WEAR


# Case 3: Missing lane-line segment (fragmentation & loss of continuity)
def test_case_03_missing_lane_line_segment(engine):
    tax = MarkingTaxonomy.LANE_LINE
    obs = engine.process_marking(
        marking_id="LANE_03", frame_idx=3, taxonomy=tax, bbox=(100, 200, 120, 500),
        paint_coverage_ratio=0.20, paint_contrast_ratio=0.35, continuity_ratio=0.30, fragmentation_count=5
    )
    assert obs.condition in [MarkingCondition.FADED, MarkingCondition.SEVERELY_DEGRADED]


# Case 4: Turn arrow (must NOT be crosswalk)
def test_case_04_turn_arrow(engine):
    tax = engine.classify_marking_type(is_arrow=True, shape_type_hint="left_arrow")
    assert tax == MarkingTaxonomy.TURN_ARROW
    assert tax != MarkingTaxonomy.ZEBRA_CROSSING

    obs = engine.process_marking(
        marking_id="ARR_01", frame_idx=4, taxonomy=tax, bbox=(300, 400, 380, 500),
        paint_coverage_ratio=0.85, paint_contrast_ratio=0.70
    )
    assert obs.condition == MarkingCondition.GOOD


# Case 5: Faded turn arrow
def test_case_05_faded_turn_arrow(engine):
    tax = MarkingTaxonomy.TURN_ARROW
    obs = engine.process_marking(
        marking_id="ARR_02", frame_idx=5, taxonomy=tax, bbox=(300, 400, 380, 500),
        paint_coverage_ratio=0.18, paint_contrast_ratio=0.32
    )
    assert obs.condition in [MarkingCondition.FADED, MarkingCondition.MODERATE_WEAR]


# Case 6: Hatch marking
def test_case_06_hatch_marking(engine):
    tax = engine.classify_marking_type(shape_type_hint="hatch")
    assert tax == MarkingTaxonomy.HATCH_MARKING
    assert tax != MarkingTaxonomy.ZEBRA_CROSSING


# Case 7: Good crosswalk (multiple transverse bars)
def test_case_07_good_crosswalk(engine):
    tax = engine.classify_marking_type(is_transverse_group=True, stripe_count=8)
    assert tax == MarkingTaxonomy.ZEBRA_CROSSING

    obs = engine.process_marking(
        marking_id="ZEBRA_01", frame_idx=7, taxonomy=tax, bbox=(150, 450, 750, 600),
        paint_coverage_ratio=0.78, paint_contrast_ratio=0.80
    )
    assert obs.condition == MarkingCondition.GOOD


# Case 8: Faded crosswalk
def test_case_08_faded_crosswalk(engine):
    tax = MarkingTaxonomy.ZEBRA_CROSSING
    obs = engine.process_marking(
        marking_id="ZEBRA_02", frame_idx=8, taxonomy=tax, bbox=(150, 450, 750, 600),
        paint_coverage_ratio=0.10, paint_contrast_ratio=0.20
    )
    assert obs.condition in [MarkingCondition.FADED, MarkingCondition.SEVERELY_DEGRADED]


# Case 9: No crosswalk
def test_case_09_no_crosswalk(engine):
    tax = engine.classify_marking_type(is_transverse_group=False, stripe_count=0)
    assert tax != MarkingTaxonomy.ZEBRA_CROSSING


# Case 10: Bus-lane marking (bus-lane stencil != crosswalk)
def test_case_10_bus_lane_marking_not_crosswalk(engine):
    tax = engine.classify_marking_type(is_bus_stencil=True, raw_text_hint="BUS LANE")
    assert tax == MarkingTaxonomy.BUS_LANE_MARKING
    assert tax != MarkingTaxonomy.ZEBRA_CROSSING


# Case 11: Road text (road text != crosswalk)
def test_case_11_road_text_not_crosswalk(engine):
    tax = engine.classify_marking_type(raw_text_hint="SLOW")
    assert tax == MarkingTaxonomy.ROAD_TEXT
    assert tax != MarkingTaxonomy.ZEBRA_CROSSING


# Case 12: Occluded marking (occluded != faded)
def test_case_12_occluded_not_faded(engine):
    tax = MarkingTaxonomy.ZEBRA_CROSSING
    # Vehicle sitting on crosswalk causing 60% occlusion
    obs = engine.process_marking(
        marking_id="ZEBRA_OCC", frame_idx=12, taxonomy=tax, bbox=(150, 450, 750, 600),
        paint_coverage_ratio=0.80, paint_contrast_ratio=0.75, occlusion_ratio=0.60
    )
    # MUST be OBSCURED, strictly NOT FADED
    assert obs.condition == MarkingCondition.OBSCURED
    assert obs.condition != MarkingCondition.FADED


# Case 13: Wet/glare scene (glare != faded)
def test_case_13_glare_not_faded(engine):
    tax = MarkingTaxonomy.LANE_LINE
    # Glare reduces scene visibility score to 0.45
    obs = engine.process_marking(
        marking_id="LANE_GLARE", frame_idx=13, taxonomy=tax, bbox=(100, 200, 120, 500),
        paint_coverage_ratio=0.80, paint_contrast_ratio=0.70, visibility_score=0.45
    )
    # MUST be LOW_VISIBILITY, strictly NOT FADED
    assert obs.condition == MarkingCondition.LOW_VISIBILITY
    assert obs.condition != MarkingCondition.FADED


# Case 14: Shadowed road (shadow != faded)
def test_case_14_shadow_not_faded(engine):
    tax = MarkingTaxonomy.LANE_LINE
    obs = engine.process_marking(
        marking_id="LANE_SHADOW", frame_idx=14, taxonomy=tax, bbox=(100, 200, 120, 500),
        paint_coverage_ratio=0.80, paint_contrast_ratio=0.70, occlusion_ratio=0.45
    )
    assert obs.condition == MarkingCondition.OBSCURED
    assert obs.condition != MarkingCondition.FADED


# Road segment aggregation test across 100m segment
def test_segment_health_aggregation(engine):
    seg_id = "SEG_100M_ALPHA"
    engine.process_marking("L1", 1, MarkingTaxonomy.LANE_LINE, (100, 200, 120, 500), paint_coverage_ratio=0.85, paint_contrast_ratio=0.70, segment_id=seg_id)
    engine.process_marking("E1", 1, MarkingTaxonomy.EDGE_LINE, (50, 200, 70, 500), paint_coverage_ratio=0.75, paint_contrast_ratio=0.65, segment_id=seg_id)
    engine.process_marking("A1", 1, MarkingTaxonomy.TURN_ARROW, (300, 400, 380, 500), paint_coverage_ratio=0.80, paint_contrast_ratio=0.70, segment_id=seg_id)
    engine.process_marking("Z1", 1, MarkingTaxonomy.ZEBRA_CROSSING, (150, 450, 750, 600), paint_coverage_ratio=0.10, paint_contrast_ratio=0.25, segment_id=seg_id)

    seg_health = engine.aggregate_segment_health(seg_id, station_start_m=0.0, station_end_m=100.0)
    assert seg_health.lane_lines_condition == MarkingCondition.GOOD
    assert seg_health.turn_arrows_condition == MarkingCondition.GOOD
    assert seg_health.crosswalk_condition in [MarkingCondition.FADED, MarkingCondition.SEVERELY_DEGRADED]
    assert seg_health.maintenance_state == "MAINTENANCE_RECOMMENDED"
    assert seg_health.total_visible_markings == 4
