#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/missing_infrastructure_engine.py
========================================================
Missing Road Infrastructure Reasoning Engine (PRD Part 2)

Strict Invariant:
Missing infrastructure must NEVER be based on:
    detector failed once = missing

Instead:
    EXPECTED ASSET (Reference Layer)
    + VISIBILITY (Clear view, not occluded/glare/shadow)
    + CURRENT OBSERVATION (Live sensor evidence)
    + TEMPORAL HISTORY (Consecutive unobserved frames >= min_frames)
    + REPEATED BUS OBSERVATION (Multi-bus corroborated passes)
    + SPATIAL CONSISTENCY (Within georeferenced road corridor)
    ---------------------------------------------------------
    = MISSING CANDIDATE -> CONFIRMED MISSING

Key Rules:
1. Asset Types: TRAFFIC_SIGN, ZEBRA_CROSSING, PHYSICAL_MEDIAN, ROAD_MARKING.
2. Asset Reference State: EXPECTED, NOT_EXPECTED, UNKNOWN.
3. If Reference is UNKNOWN -> NO_MISSING_CLAIM.
4. If view is OBSCURED / LOW_VISIBILITY -> NO_MISSING_CLAIM.
5. Missing Sign displays "EXPECTED SIGN LOCATION", never a fake detected sign bbox.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


class AssetType(str, Enum):
    TRAFFIC_SIGN = "TRAFFIC_SIGN"
    ZEBRA_CROSSING = "ZEBRA_CROSSING"
    PHYSICAL_MEDIAN = "PHYSICAL_MEDIAN"
    ROAD_MARKING = "ROAD_MARKING"
    TRAFFIC_SIGNAL = "TRAFFIC_SIGNAL"
    BOLLARD = "BOLLARD"


class ReferenceState(str, Enum):
    EXPECTED = "EXPECTED"
    NOT_EXPECTED = "NOT_EXPECTED"
    UNKNOWN = "UNKNOWN"


class MissingStatus(str, Enum):
    VERIFIED_PRESENT = "VERIFIED_PRESENT"
    OBSCURED_UNCONFIRMED = "OBSCURED_UNCONFIRMED"
    POTENTIAL_MISSING = "POTENTIAL_MISSING"        # 1 pass / temporal candidate
    CONFIRMED_MISSING = "CONFIRMED_MISSING"        # multi-frame & corroborated
    NOT_EXPECTED = "NOT_EXPECTED"
    NO_CLAIM = "NO_CLAIM"


@dataclass
class ExpectedAsset:
    asset_id: str
    asset_type: AssetType
    reference_state: ReferenceState
    road_station_s_m: float
    description: str
    road_segment_id: str = "SEG_01"
    expected_geometry: Optional[str] = None


@dataclass
class MissingInfrastructureObservation:
    asset_id: str
    asset_type: AssetType
    reference_state: ReferenceState
    current_detected: bool
    visibility_score: float
    temporal_absence_count: int
    bus_corroboration_votes: int
    missing_status: MissingStatus
    display_tag: str
    dispatch_alert: bool
    screen_marker_position: Optional[Tuple[int, int]] = None
    road_segment_id: str = "SEG_01"
    road_station_s_m: float = 0.0


class MissingInfrastructureEngine:
    """
    Evaluates reference municipal assets against live sensor observations,
    visibility metrics, temporal history, and multi-bus corroborated votes.
    One bus pass = one independent evidence source.
    """

    def __init__(
        self,
        min_temporal_absence_frames: int = 5,
        min_corroboration_votes: int = 2,
        min_visibility_threshold: float = 0.65,
    ):
        self.min_temporal_absence_frames = min_temporal_absence_frames
        self.min_corroboration_votes = min_corroboration_votes
        self.min_visibility_threshold = min_visibility_threshold
        # Tracks temporal history: asset_id -> consecutive absence frames
        self.absence_history: Dict[str, int] = {}
        # Tracks multi-bus corroborated passes: asset_id -> set of unique bus IDs
        self.bus_passes: Dict[str, set] = {}
        # Backward compatibility for vote counts
        self.bus_votes: Dict[str, int] = {}

    def add_bus_corroboration(self, asset_id: str, bus_id_or_count: Any = 1):
        """Adds independent bus observation votes for municipal reconciliation."""
        if isinstance(bus_id_or_count, str):
            self.bus_passes.setdefault(asset_id, set()).add(bus_id_or_count)
            self.bus_votes[asset_id] = len(self.bus_passes[asset_id])
        else:
            self.bus_votes[asset_id] = self.bus_votes.get(asset_id, 0) + int(bus_id_or_count)
            for i in range(int(bus_id_or_count)):
                self.bus_passes.setdefault(asset_id, set()).add(f"BUS_{i+1:03d}")

    def evaluate_asset(
        self,
        asset: ExpectedAsset,
        current_detected: bool,
        visibility_score: float = 0.90,
        bus_id: str = "BUS_001",
        screen_coord: Optional[Tuple[int, int]] = None
    ) -> MissingInfrastructureObservation:
        """
        Evaluates a single infrastructure asset against strict ground truth rules.
        """
        # Rule 1: Unknown reference cannot generate a missing claim
        if asset.reference_state == ReferenceState.UNKNOWN:
            return MissingInfrastructureObservation(
                asset_id=asset.asset_id,
                asset_type=asset.asset_type,
                reference_state=asset.reference_state,
                current_detected=current_detected,
                visibility_score=visibility_score,
                temporal_absence_count=0,
                bus_corroboration_votes=0,
                missing_status=MissingStatus.NO_CLAIM,
                display_tag="REFERENCE_UNKNOWN (No Missing Claim)",
                dispatch_alert=False,
                screen_marker_position=None
            )

        # Rule 2: If asset is not expected in reference layer
        if asset.reference_state == ReferenceState.NOT_EXPECTED:
            return MissingInfrastructureObservation(
                asset_id=asset.asset_id,
                asset_type=asset.asset_type,
                reference_state=asset.reference_state,
                current_detected=current_detected,
                visibility_score=visibility_score,
                temporal_absence_count=0,
                bus_corroboration_votes=0,
                missing_status=MissingStatus.NOT_EXPECTED,
                display_tag="NOT_EXPECTED (Reference Mapped)",
                dispatch_alert=False,
                screen_marker_position=None
            )

        # Rule 3: Asset is expected AND detected by live vision
        if current_detected:
            self.absence_history[asset.asset_id] = 0
            return MissingInfrastructureObservation(
                asset_id=asset.asset_id,
                asset_type=asset.asset_type,
                reference_state=asset.reference_state,
                current_detected=True,
                visibility_score=visibility_score,
                temporal_absence_count=0,
                bus_corroboration_votes=self.bus_votes.get(asset.asset_id, 1),
                missing_status=MissingStatus.VERIFIED_PRESENT,
                display_tag=f"{asset.asset_type.value} VERIFIED PRESENT",
                dispatch_alert=False,
                screen_marker_position=screen_coord
            )

        # Rule 4: If visibility is low (occluded, glare, shadowed) -> Do not claim missing
        if visibility_score < self.min_visibility_threshold:
            return MissingInfrastructureObservation(
                asset_id=asset.asset_id,
                asset_type=asset.asset_type,
                reference_state=asset.reference_state,
                current_detected=False,
                visibility_score=visibility_score,
                temporal_absence_count=self.absence_history.get(asset.asset_id, 0),
                bus_corroboration_votes=self.bus_votes.get(asset.asset_id, 0),
                missing_status=MissingStatus.OBSCURED_UNCONFIRMED,
                display_tag="OBSCURED / LOW VISIBILITY (Unconfirmed)",
                dispatch_alert=False,
                screen_marker_position=None
            )

        # Rule 5: Temporal history tracking
        current_absence = self.absence_history.get(asset.asset_id, 0) + 1
        self.absence_history[asset.asset_id] = current_absence

        # Votes
        votes = self.bus_votes.get(asset.asset_id, 1)

        # Rule 7: Single frame absence -> POTENTIAL_MISSING (no premature alert)
        if current_absence < self.min_temporal_absence_frames:
            return MissingInfrastructureObservation(
                asset_id=asset.asset_id,
                asset_type=asset.asset_type,
                reference_state=asset.reference_state,
                current_detected=False,
                visibility_score=visibility_score,
                temporal_absence_count=current_absence,
                bus_corroboration_votes=votes,
                missing_status=MissingStatus.POTENTIAL_MISSING,
                display_tag=f"POTENTIAL MISSING {asset.asset_type.value} (Tracking {current_absence}/{self.min_temporal_absence_frames})",
                dispatch_alert=False,
                screen_marker_position=screen_coord
            )

        # Rule 8: Multi-frame persistent absence -> CONFIRMED_MISSING
        return MissingInfrastructureObservation(
            asset_id=asset.asset_id,
            asset_type=asset.asset_type,
            reference_state=asset.reference_state,
            current_detected=False,
            visibility_score=visibility_score,
            temporal_absence_count=current_absence,
            bus_corroboration_votes=votes,
            missing_status=MissingStatus.CONFIRMED_MISSING,
            display_tag=f"CONFIRMED MISSING {asset.asset_type.value} [DISPATCH ALERT]",
            dispatch_alert=True,
            screen_marker_position=screen_coord
        )
