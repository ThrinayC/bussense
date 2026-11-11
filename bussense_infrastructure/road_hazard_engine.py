#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/road_hazard_engine.py
=============================================
General Road Hazards & Geometric Unknown Obstruction Engine (Section 17)

Supported MVP Classes:
- DEBRIS
- CONSTRUCTION_BARRIER
- STALLED_VEHICLE
- ROAD_BLOCKAGE
- FALLEN_OBJECT

Geometric Unknown Obstruction:
- Tracked 3D obstacle inside the drivable ego-corridor that does not match
  known vehicle or VRU categories is reported as UNKNOWN_OBSTRUCTION.
- No learned open-world hallucinations.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class RoadHazardObservation:
    hazard_id: str
    hazard_type: str  # DEBRIS, CONSTRUCTION_BARRIER, STALLED_VEHICLE, ROAD_BLOCKAGE, FALLEN_OBJECT, UNKNOWN_OBSTRUCTION
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance_m: float
    in_ego_corridor: bool
    lateral_offset_m: float
    height_3d_m: Optional[float] = None
    is_geometric_unknown: bool = False


class RoadHazardEngine:
    """
    Evaluates supported road hazard classes and geometric obstructions.
    """

    SUPPORTED_CLASSES = [
        "DEBRIS",
        "CONSTRUCTION_BARRIER",
        "STALLED_VEHICLE",
        "ROAD_BLOCKAGE",
        "FALLEN_OBJECT"
    ]

    def __init__(
        self,
        corridor_half_width_m: float = 1.8,
        min_obstruction_height_m: float = 0.20,
    ):
        self.corridor_half_width_m = corridor_half_width_m
        self.min_obstruction_height_m = min_obstruction_height_m

    def evaluate_candidate(
        self,
        category: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        position_3d: Tuple[float, float, float],  # (x, y, z) in camera coordinates
        speed_mps: float = 0.0
    ) -> Optional[RoadHazardObservation]:
        """
        Evaluates detected object or unknown 3D obstacle against road hazard criteria.
        """
        x, y, z = position_3d
        in_corridor = (abs(x) <= self.corridor_half_width_m) and (0.5 <= z <= 40.0)

        # Check for stalled vehicle (stationary in corridor)
        if category in ["car", "truck", "bus"] and in_corridor and speed_mps < 0.3 and z < 25.0:
            return RoadHazardObservation(
                hazard_id=f"HAZ-STALLED-{int(z*10)}",
                hazard_type="STALLED_VEHICLE",
                confidence=confidence,
                bbox=bbox,
                distance_m=round(z, 2),
                in_ego_corridor=in_corridor,
                lateral_offset_m=round(x, 2),
                height_3d_m=1.5
            )

        # Check for known supported hazard classes
        cat_upper = category.upper()
        if cat_upper in self.SUPPORTED_CLASSES:
            return RoadHazardObservation(
                hazard_id=f"HAZ-{cat_upper}-{int(z*10)}",
                hazard_type=cat_upper,
                confidence=confidence,
                bbox=bbox,
                distance_m=round(z, 2),
                in_ego_corridor=in_corridor,
                lateral_offset_m=round(x, 2),
                height_3d_m=0.5
            )

        # Geometric Unknown Obstruction
        if in_corridor and category not in ["car", "truck", "bus", "pedestrian", "cyclist"]:
            # Object occupying drivable path
            return RoadHazardObservation(
                hazard_id=f"HAZ-UNKNOWN-{int(z*10)}",
                hazard_type="UNKNOWN_OBSTRUCTION",
                confidence=float(np.clip(confidence, 0.60, 0.85)),
                bbox=bbox,
                distance_m=round(z, 2),
                in_ego_corridor=True,
                lateral_offset_m=round(x, 2),
                height_3d_m=0.35,
                is_geometric_unknown=True
            )

        return None
