#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_divider_taxonomy_and_metrology.py
============================================
Comprehensive test suite verifying Part 1 of the BusSense Final Infrastructure Fix:
- Strict 10-class taxonomy (PHYSICAL_MEDIAN, JERSEY_BARRIER, RAISED_DIVIDER, TRAFFIC_ISLAND,
  CURB_ONLY, ROAD_EDGE, GRASS_EDGE, LANE_MARKING, WALL_EDGE, GUARDRAIL, NO_DIVIDER)
- Carriageway separation constraints: Curbs, road edges, roadside walls, roadside guardrails,
  and lane markings produce NO DIVIDER event and ZERO lines
- Population-based 3D height metrology (P10, P50, P90, median, MAD, IQR)
- Instability filtering (unstable point distribution produces HEIGHT: N/A)
- Non-persistence guarantee: measurement validity strictly does not persist across frames without stereo
- 10 Required Test Cases (A through J) per Section 12
"""

import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.road_divider_engine import (
    RoadDividerEngine,
    DividerTaxonomy,
    VALID_ROAD_DIVIDER_CLASSES
)


@pytest.fixture
def engine():
    return RoadDividerEngine(min_physical_height_cm=12.0, max_physical_height_cm=100.0, min_point_count=10)


@pytest.fixture
def synthetic_road_plane():
    # Normal pointing up: (0, -1, 0), road surface at y=1.5m below camera
    return (0.0, -1.0, 0.0, 1.5)


def test_case_a_physical_concrete_median(engine, synthetic_road_plane):
    """CASE A: Physical concrete median -> PHYSICAL_MEDIAN -> real height measured from stereo."""
    taxonomy = engine.classify_structure(
        has_visible_vertical_structure=True,
        is_outer_road_edge=False,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=True,
        structure_type_hint=None
    )
    assert taxonomy == DividerTaxonomy.PHYSICAL_MEDIAN
    assert taxonomy in VALID_ROAD_DIVIDER_CLASSES

    np.random.seed(42)
    y_vals = 1.50 - (0.28 + np.random.normal(0, 0.015, 30))
    pts_3d = np.column_stack([np.full(30, -0.5), y_vals, np.linspace(10, 20, 30)])

    obs = engine.process_frame(
        frame_idx=100,
        timestamp=10.0,
        taxonomy=taxonomy,
        points_3d=pts_3d,
        road_plane=synthetic_road_plane,
        screen_polygon=[(200, 300), (250, 600)]
    )

    assert obs.is_road_divider is True
    assert obs.divider_present is True
    assert obs.height_metrology.is_valid is True
    assert 26.0 <= obs.height_metrology.p50_cm <= 30.0
    assert obs.height_metrology.p10_cm <= obs.height_metrology.p50_cm <= obs.height_metrology.p90_cm
    assert obs.screen_polygon is not None
    assert obs.localization is not None
    assert obs.localization.is_valid_localization is True


def test_case_b_jersey_barrier(engine, synthetic_road_plane):
    """CASE B: Jersey barrier -> JERSEY_BARRIER -> real height measured from stereo."""
    taxonomy = engine.classify_structure(
        has_visible_vertical_structure=True,
        is_outer_road_edge=False,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=True,
        structure_type_hint="jersey_barrier"
    )
    assert taxonomy == DividerTaxonomy.JERSEY_BARRIER
    assert taxonomy in VALID_ROAD_DIVIDER_CLASSES

    y_vals = 1.50 - (0.81 + np.random.normal(0, 0.02, 25))
    pts_3d = np.column_stack([np.full(25, -1.0), y_vals, np.linspace(8, 18, 25)])

    obs = engine.process_frame(
        frame_idx=101,
        timestamp=10.1,
        taxonomy=taxonomy,
        points_3d=pts_3d,
        road_plane=synthetic_road_plane
    )

    assert obs.is_road_divider is True
    assert obs.height_metrology.is_valid is True
    assert 78.0 <= obs.height_metrology.p50_cm <= 84.0


def test_case_c_raised_divider(engine, synthetic_road_plane):
    """CASE C: Raised divider -> RAISED_DIVIDER -> qualifies as Road Divider."""
    taxonomy = engine.classify_structure(
        has_visible_vertical_structure=True,
        is_outer_road_edge=False,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=True,
        structure_type_hint="raised_divider"
    )
    assert taxonomy == DividerTaxonomy.RAISED_DIVIDER
    assert taxonomy in VALID_ROAD_DIVIDER_CLASSES


def test_case_d_traffic_island(engine, synthetic_road_plane):
    """CASE D: Traffic island -> TRAFFIC_ISLAND -> qualifies as Road Divider."""
    taxonomy = engine.classify_structure(
        has_visible_vertical_structure=True,
        is_outer_road_edge=False,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=True,
        structure_type_hint="traffic_island"
    )
    assert taxonomy == DividerTaxonomy.TRAFFIC_ISLAND
    assert taxonomy in VALID_ROAD_DIVIDER_CLASSES


def test_case_e_roadside_curb(engine, synthetic_road_plane):
    """CASE E: Ordinary roadside curb -> CURB_ONLY -> NO DIVIDER event, zero lines."""
    taxonomy = engine.classify_structure(
        has_visible_vertical_structure=True,
        is_outer_road_edge=True,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=False
    )
    assert taxonomy == DividerTaxonomy.CURB_ONLY
    assert taxonomy not in VALID_ROAD_DIVIDER_CLASSES

    obs = engine.process_frame(
        frame_idx=102,
        timestamp=10.2,
        taxonomy=taxonomy,
        points_3d=np.zeros((20, 3)),
        road_plane=synthetic_road_plane,
        screen_polygon=[(50, 400), (80, 700)]
    )

    assert obs.is_road_divider is False
    assert obs.divider_present is False
    assert obs.screen_polygon is None  # Zero lines drawn
    assert obs.localization is None
    assert obs.height_metrology.is_valid is False
    assert obs.height_metrology.summary_text() == "N/A"


def test_case_f_road_edge(engine):
    """CASE F: Road edge -> ROAD_EDGE -> NO DIVIDER event."""
    taxonomy = engine.classify_structure(
        has_visible_vertical_structure=False,
        is_outer_road_edge=True,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=False
    )
    assert taxonomy == DividerTaxonomy.ROAD_EDGE
    assert taxonomy not in VALID_ROAD_DIVIDER_CLASSES


def test_case_g_lane_marking(engine):
    """CASE G: Painted lane marking -> LANE_MARKING -> NO DIVIDER event."""
    taxonomy = engine.classify_structure(
        has_visible_vertical_structure=False,
        is_outer_road_edge=False,
        is_flat_paint=True,
        is_grass_adjacent=False,
        separates_opposing_carriageways=True
    )
    assert taxonomy == DividerTaxonomy.LANE_MARKING
    assert taxonomy not in VALID_ROAD_DIVIDER_CLASSES


def test_case_h_guardrail_and_wall(engine):
    """CASE H: Roadside guardrail and wall -> GUARDRAIL / WALL_EDGE -> NOT a median."""
    tax_guardrail = engine.classify_structure(
        has_visible_vertical_structure=True,
        is_outer_road_edge=True,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=False,
        is_guardrail=True
    )
    assert tax_guardrail == DividerTaxonomy.GUARDRAIL
    assert tax_guardrail not in VALID_ROAD_DIVIDER_CLASSES

    tax_wall = engine.classify_structure(
        has_visible_vertical_structure=True,
        is_outer_road_edge=True,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=False,
        is_wall=True
    )
    assert tax_wall == DividerTaxonomy.WALL_EDGE
    assert tax_wall not in VALID_ROAD_DIVIDER_CLASSES


def test_case_i_no_divider(engine):
    """CASE I: No separator -> NO_DIVIDER -> NO DIVIDER event."""
    taxonomy = engine.classify_structure(
        has_visible_vertical_structure=False,
        is_outer_road_edge=False,
        is_flat_paint=False,
        is_grass_adjacent=False,
        separates_opposing_carriageways=False
    )
    assert taxonomy == DividerTaxonomy.NO_DIVIDER
    assert taxonomy not in VALID_ROAD_DIVIDER_CLASSES


def test_case_j_invalid_stereo_and_non_persistence(engine, synthetic_road_plane):
    """
    CASE J: Section 8, 10 & 12 Non-Persistence and Invalid Stereo:
    - Frame 100: valid stereo -> HEIGHT = measured
    - Frame 101: valid stereo -> HEIGHT = measured
    - Frame 102: stereo invalid -> HEIGHT = N/A
    Measurement MUST NOT persist when stereo is invalid!
    """
    taxonomy = DividerTaxonomy.PHYSICAL_MEDIAN

    # Frame 100: valid stereo
    y_vals = 1.50 - (0.28 + np.random.normal(0, 0.01, 20))
    pts_100 = np.column_stack([np.full(20, -0.5), y_vals, np.linspace(10, 20, 20)])
    obs_100 = engine.process_frame(
        frame_idx=100, timestamp=10.0, taxonomy=taxonomy, points_3d=pts_100, road_plane=synthetic_road_plane
    )
    assert obs_100.height_metrology.is_valid is True
    assert 26.0 <= obs_100.height_metrology.p50_cm <= 30.0

    # Frame 101: valid stereo
    y_vals_101 = 1.50 - (0.29 + np.random.normal(0, 0.01, 20))
    pts_101 = np.column_stack([np.full(20, -0.5), y_vals_101, np.linspace(10, 20, 20)])
    obs_101 = engine.process_frame(
        frame_idx=101, timestamp=10.1, taxonomy=taxonomy, points_3d=pts_101, road_plane=synthetic_road_plane
    )
    assert obs_101.height_metrology.is_valid is True

    # Frame 102: stereo drops out (points_3d = None)
    obs_102 = engine.process_frame(
        frame_idx=102, timestamp=10.2, taxonomy=taxonomy, points_3d=None, road_plane=synthetic_road_plane
    )
    # MUST strictly be N/A. No caching from Frame 101!
    assert obs_102.height_metrology.is_valid is False
    assert obs_102.height_metrology.summary_text() == "N/A"
    assert obs_102.stereo_status == "N/A (Monocular / No Disparity)"

    # Frame 103: unstable / noisy points (high MAD / IQR)
    noisy_pts = np.column_stack([
        np.full(20, -0.5),
        np.linspace(0.2, 1.4, 20),  # Spread from 10cm to 130cm
        np.linspace(10, 20, 20)
    ])
    obs_103 = engine.process_frame(
        frame_idx=103, timestamp=10.3, taxonomy=taxonomy, points_3d=noisy_pts, road_plane=synthetic_road_plane
    )
    # High spread must be rejected as unstable
    assert obs_103.height_metrology.is_valid is False
    assert obs_103.height_metrology.summary_text() == "N/A"


def test_case_k_landscaped_median(engine):
    """CASE K: Landscaped median separating carriageways vs outer grass edge."""
    # Landscaped median physically separating opposing travel lanes -> LANDSCAPED_MEDIAN
    median_tax = engine.classify_structure(
        has_visible_vertical_structure=False,
        is_outer_road_edge=False,
        is_flat_paint=False,
        is_grass_adjacent=True,
        separates_opposing_carriageways=True,
        is_vegetation=True
    )
    assert median_tax == DividerTaxonomy.LANDSCAPED_MEDIAN
    assert median_tax in VALID_ROAD_DIVIDER_CLASSES

    # Outer roadside grass on road boundary -> GRASS_EDGE (hard rejected)
    edge_tax = engine.classify_structure(
        has_visible_vertical_structure=False,
        is_outer_road_edge=True,
        is_flat_paint=False,
        is_grass_adjacent=True,
        separates_opposing_carriageways=False,
        is_vegetation=True
    )
    assert edge_tax == DividerTaxonomy.GRASS_EDGE
    assert edge_tax not in VALID_ROAD_DIVIDER_CLASSES

