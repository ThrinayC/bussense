#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/road_marking_health_engine.py
=====================================================
Road Marking & Lane Guidance Health Engine (PRD Part 3 & 4)

Replaces narrow "zebra paint %" with comprehensive, holistic road marking intelligence.

Key Invariants:
1. Strict Marking Taxonomy:
   LANE_LINE, EDGE_LINE, CENTER_LINE, STOP_LINE, TURN_ARROW, DIRECTION_ARROW,
   HATCH_MARKING, CHEVRON, ZEBRA_CROSSING, BUS_LANE_MARKING, ROAD_TEXT, OTHER_MARKING, UNKNOWN.
2. Anti-False Crosswalk Invariant:
   A crosswalk requires multiple transverse bars (>= 4) grouped together spanning the travel corridor.
   - BUS_LANE_MARKING ("BUS LANE") is NOT a crosswalk.
   - Lane dividers / center lines are NOT crosswalks.
   - Turn arrows / straight arrows are NOT crosswalks.
   - Road text ("SLOW", "STOP") is NOT a crosswalk.
3. Visibility-Aware Condition:
   Occlusion (vehicles, shadows, glare) produces OBSCURED or LOW_VISIBILITY, NEVER FADED.
4. Per-Marking Health:
   Continuity, fragmentation count, paint coverage ratio, paint contrast relative to pavement, edge integrity.
5. Road Segment Spatial Aggregation:
   Aggregates observations across road stations (e.g. 100m segment) for municipal work order dispatch.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import cv2


class MarkingTaxonomy(str, Enum):
    LANE_LINE = "LANE_LINE"
    EDGE_LINE = "EDGE_LINE"
    CENTER_LINE = "CENTER_LINE"
    STOP_LINE = "STOP_LINE"
    TURN_ARROW = "TURN_ARROW"
    DIRECTION_ARROW = "DIRECTION_ARROW"
    HATCH_MARKING = "HATCH_MARKING"
    CHEVRON = "CHEVRON"
    ZEBRA_CROSSING = "ZEBRA_CROSSING"
    BUS_LANE_MARKING = "BUS_LANE_MARKING"
    ROAD_TEXT = "ROAD_TEXT"
    OTHER_MARKING = "OTHER_MARKING"
    UNKNOWN = "UNKNOWN"


class MarkingCondition(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    MODERATE_WEAR = "MODERATE_WEAR"
    FADED = "FADED"
    SEVERELY_DEGRADED = "SEVERELY_DEGRADED"
    OBSCURED = "OBSCURED"
    LOW_VISIBILITY = "LOW_VISIBILITY"
    NOT_OBSERVED = "NOT_OBSERVED"


@dataclass
class MarkingObservation:
    marking_id: str
    frame_idx: int
    taxonomy: MarkingTaxonomy
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    polygon: Optional[List[Tuple[int, int]]] = None
    visible_area_px: int = 0
    paint_coverage_ratio: float = 0.0      # 0.0 - 1.0
    paint_contrast_ratio: float = 0.0      # relative to asphalt
    continuity_ratio: float = 1.0          # for lines: continuous fraction
    fragmentation_count: int = 0           # breaks / interruptions
    edge_integrity: float = 1.0            # boundary sharpness (0.0 - 1.0)
    occlusion_ratio: float = 0.0           # vehicle/shadow cover (0.0 - 1.0)
    visibility_score: float = 1.0
    condition: MarkingCondition = MarkingCondition.GOOD
    longitudinal_s_m: float = 0.0          # road station distance
    confidence: float = 0.90

    def summary_text(self) -> str:
        return f"{self.taxonomy.value} [{self.condition.value}] Cov:{self.paint_coverage_ratio*100:.1f}% Cont:{self.continuity_ratio*100:.0f}%"


@dataclass
class RoadSegmentMarkingHealth:
    segment_id: str
    station_start_m: float
    station_end_m: float
    lane_lines_condition: MarkingCondition
    edge_lines_condition: MarkingCondition
    turn_arrows_condition: MarkingCondition
    crosswalk_condition: MarkingCondition
    hatch_condition: MarkingCondition
    total_visible_markings: int
    expected_markings_count: int
    maintenance_state: str  # "GOOD", "ACCEPTABLE", "MAINTENANCE_RECOMMENDED", "CRITICAL_ACTION_REQUIRED"

    def summary_lines(self) -> List[str]:
        return [
            f"Segment {self.segment_id} ({self.station_start_m:.0f}m - {self.station_end_m:.0f}m):",
            f"  Lane Lines:    {self.lane_lines_condition.value}",
            f"  Edge Lines:    {self.edge_lines_condition.value}",
            f"  Turn Arrows:   {self.turn_arrows_condition.value}",
            f"  Crosswalk:     {self.crosswalk_condition.value}",
            f"  Hatch:         {self.hatch_condition.value}",
            f"  Markings Seen: {self.total_visible_markings}/{self.expected_markings_count}",
            f"  Maintenance:   {self.maintenance_state}",
        ]


class RoadMarkingHealthEngine:
    """
    Evaluates individual road marking instances and aggregates road segment health.
    Enforces strict distinction between crosswalks, bus-lane markings, arrows, and line markings.
    """

    def __init__(
        self,
        min_crosswalk_bars: int = 4,
        good_paint_threshold: float = 0.60,
        moderate_paint_threshold: float = 0.35,
        faded_paint_threshold: float = 0.15,
        contrast_threshold: float = 0.30,
        occlusion_threshold: float = 0.40,
    ):
        self.min_crosswalk_bars = min_crosswalk_bars
        self.good_paint_threshold = good_paint_threshold
        self.moderate_paint_threshold = moderate_paint_threshold
        self.faded_paint_threshold = faded_paint_threshold
        self.contrast_threshold = contrast_threshold
        self.occlusion_threshold = occlusion_threshold
        # Spatial registry: segment_id -> list of observations
        self.segment_observations: Dict[str, List[MarkingObservation]] = {}

    def classify_marking_type(
        self,
        raw_text_hint: Optional[str] = None,
        shape_type_hint: Optional[str] = None,
        stripe_count: int = 0,
        is_transverse_group: bool = False,
        is_longitudinal_line: bool = False,
        is_arrow: bool = False,
        is_bus_stencil: bool = False,
    ) -> MarkingTaxonomy:
        """
        Anti-False Crosswalk Classification (Section 21).
        Strictly prevents calling bus-lane stencils, turn arrows, or lane lines crosswalks.
        """
        # Rule 1: Stencil text like "BUS LANE" is strictly BUS_LANE_MARKING
        if is_bus_stencil or (raw_text_hint and "BUS" in raw_text_hint.upper()):
            return MarkingTaxonomy.BUS_LANE_MARKING

        # Rule 2: General road text ("SLOW", "STOP")
        if raw_text_hint and ("SLOW" in raw_text_hint.upper() or "STOP" in raw_text_hint.upper()):
            if "STOP" in raw_text_hint.upper() and not is_transverse_group:
                return MarkingTaxonomy.STOP_LINE
            return MarkingTaxonomy.ROAD_TEXT

        # Rule 3: Arrows
        if is_arrow or shape_type_hint in ["left_arrow", "right_arrow", "turn_arrow"]:
            return MarkingTaxonomy.TURN_ARROW
        elif shape_type_hint in ["straight_arrow", "direction_arrow"]:
            return MarkingTaxonomy.DIRECTION_ARROW

        # Rule 4: Crosswalk requires multiple transverse bars grouped together
        if is_transverse_group and stripe_count >= self.min_crosswalk_bars:
            return MarkingTaxonomy.ZEBRA_CROSSING

        # Rule 5: Longitudinal line markings
        if is_longitudinal_line:
            if shape_type_hint == "edge_line":
                return MarkingTaxonomy.EDGE_LINE
            elif shape_type_hint == "center_line":
                return MarkingTaxonomy.CENTER_LINE
            else:
                return MarkingTaxonomy.LANE_LINE

        # Rule 6: Hatch / Chevron markings
        if shape_type_hint in ["hatch", "hatch_marking"]:
            return MarkingTaxonomy.HATCH_MARKING
        elif shape_type_hint in ["chevron", "chevron_marking"]:
            return MarkingTaxonomy.CHEVRON

        return MarkingTaxonomy.OTHER_MARKING

    def evaluate_marking_condition(
        self,
        paint_coverage_ratio: float,
        paint_contrast_ratio: float,
        continuity_ratio: Optional[float] = None,
        fragmentation_count: int = 0,
        occlusion_ratio: float = 0.0,
        visibility_score: float = 1.0,
    ) -> MarkingCondition:
        """
        Calculates health condition with strict visibility awareness (Section 30).
        """
        # Rule 1: Occluded or low visibility cannot be branded as faded
        if occlusion_ratio >= self.occlusion_threshold:
            return MarkingCondition.OBSCURED
        if visibility_score < 0.60:
            return MarkingCondition.LOW_VISIBILITY

        # Rule 2: Severe contrast degradation (indistinguishable from asphalt)
        if paint_contrast_ratio < self.contrast_threshold * 0.5:
            return MarkingCondition.SEVERELY_DEGRADED

        # Rule 3: Paint coverage & continuity scoring
        if continuity_ratio is not None:
            effective_score = paint_coverage_ratio * 0.6 + continuity_ratio * 0.4
        else:
            effective_score = paint_coverage_ratio

        if fragmentation_count >= 4:
            effective_score *= 0.80

        if effective_score >= self.good_paint_threshold:
            return MarkingCondition.GOOD
        elif effective_score >= self.moderate_paint_threshold:
            return MarkingCondition.ACCEPTABLE
        elif effective_score >= self.faded_paint_threshold:
            return MarkingCondition.MODERATE_WEAR if effective_score >= 0.25 else MarkingCondition.FADED
        else:
            return MarkingCondition.SEVERELY_DEGRADED

    def process_marking(
        self,
        marking_id: str,
        frame_idx: int,
        taxonomy: MarkingTaxonomy,
        bbox: Tuple[int, int, int, int],
        polygon: Optional[List[Tuple[int, int]]] = None,
        paint_coverage_ratio: float = 0.85,
        paint_contrast_ratio: float = 0.70,
        continuity_ratio: float = 1.0,
        fragmentation_count: int = 0,
        edge_integrity: float = 0.90,
        occlusion_ratio: float = 0.0,
        visibility_score: float = 1.0,
        longitudinal_s_m: float = 0.0,
        segment_id: str = "SEG_01",
    ) -> MarkingObservation:
        """
        Creates a validated MarkingObservation and registers it with the road segment.
        """
        is_line = taxonomy in [MarkingTaxonomy.LANE_LINE, MarkingTaxonomy.EDGE_LINE, MarkingTaxonomy.CENTER_LINE]
        eff_continuity = continuity_ratio if is_line else None

        condition = self.evaluate_marking_condition(
            paint_coverage_ratio=paint_coverage_ratio,
            paint_contrast_ratio=paint_contrast_ratio,
            continuity_ratio=eff_continuity,
            fragmentation_count=fragmentation_count,
            occlusion_ratio=occlusion_ratio,
            visibility_score=visibility_score
        )

        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        visible_area_px = int(w * h * (1.0 - occlusion_ratio))

        obs = MarkingObservation(
            marking_id=marking_id,
            frame_idx=frame_idx,
            taxonomy=taxonomy,
            bbox=bbox,
            polygon=polygon,
            visible_area_px=visible_area_px,
            paint_coverage_ratio=round(paint_coverage_ratio, 3),
            paint_contrast_ratio=round(paint_contrast_ratio, 3),
            continuity_ratio=round(continuity_ratio, 3),
            fragmentation_count=fragmentation_count,
            edge_integrity=round(edge_integrity, 3),
            occlusion_ratio=round(occlusion_ratio, 3),
            visibility_score=round(visibility_score, 3),
            condition=condition,
            longitudinal_s_m=round(longitudinal_s_m, 1)
        )

        if segment_id not in self.segment_observations:
            self.segment_observations[segment_id] = []
        self.segment_observations[segment_id].append(obs)

        return obs

    def aggregate_segment_health(
        self,
        segment_id: str,
        station_start_m: float = 0.0,
        station_end_m: float = 100.0,
        expected_markings_count: int = 10,
    ) -> RoadSegmentMarkingHealth:
        """
        Aggregates individual markings along a 100m road section (Section 27 & 28).
        """
        obs_list = self.segment_observations.get(segment_id, [])

        # Categorize
        lane_conds = [o.condition for o in obs_list if o.taxonomy in [MarkingTaxonomy.LANE_LINE, MarkingTaxonomy.CENTER_LINE]]
        edge_conds = [o.condition for o in obs_list if o.taxonomy == MarkingTaxonomy.EDGE_LINE]
        arrow_conds = [o.condition for o in obs_list if o.taxonomy in [MarkingTaxonomy.TURN_ARROW, MarkingTaxonomy.DIRECTION_ARROW]]
        zebra_conds = [o.condition for o in obs_list if o.taxonomy == MarkingTaxonomy.ZEBRA_CROSSING]
        hatch_conds = [o.condition for o in obs_list if o.taxonomy in [MarkingTaxonomy.HATCH_MARKING, MarkingTaxonomy.CHEVRON]]

        def worst_condition(cond_list: List[MarkingCondition], default=MarkingCondition.GOOD) -> MarkingCondition:
            if not cond_list:
                return default
            order = [
                MarkingCondition.SEVERELY_DEGRADED,
                MarkingCondition.FADED,
                MarkingCondition.MODERATE_WEAR,
                MarkingCondition.ACCEPTABLE,
                MarkingCondition.GOOD,
                MarkingCondition.OBSCURED,
                MarkingCondition.LOW_VISIBILITY,
                MarkingCondition.NOT_OBSERVED,
            ]
            for c in order:
                if c in cond_list:
                    return c
            return default

        lane_health = worst_condition(lane_conds, MarkingCondition.GOOD)
        edge_health = worst_condition(edge_conds, MarkingCondition.ACCEPTABLE)
        arrow_health = worst_condition(arrow_conds, MarkingCondition.GOOD)
        zebra_health = worst_condition(zebra_conds, MarkingCondition.NOT_OBSERVED)
        hatch_health = worst_condition(hatch_conds, MarkingCondition.GOOD)

        # Maintenance Recommendation logic
        critical_count = sum(1 for c in [lane_health, edge_health, arrow_health, zebra_health] if c in [MarkingCondition.SEVERELY_DEGRADED, MarkingCondition.FADED])
        if critical_count >= 2:
            maint = "CRITICAL_ACTION_REQUIRED"
        elif critical_count >= 1 or lane_health == MarkingCondition.MODERATE_WEAR or edge_health == MarkingCondition.MODERATE_WEAR:
            maint = "MAINTENANCE_RECOMMENDED"
        else:
            maint = "GOOD"

        return RoadSegmentMarkingHealth(
            segment_id=segment_id,
            station_start_m=station_start_m,
            station_end_m=station_end_m,
            lane_lines_condition=lane_health,
            edge_lines_condition=edge_health,
            turn_arrows_condition=arrow_health,
            crosswalk_condition=zebra_health,
            hatch_condition=hatch_health,
            total_visible_markings=len(obs_list),
            expected_markings_count=expected_markings_count,
            maintenance_state=maint
        )
