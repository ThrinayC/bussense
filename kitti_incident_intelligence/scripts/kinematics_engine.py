"""
Kinematics Engine — Phase 4
Calculates relative and absolute kinematics, lead vehicle metrics, time headway,
and dual-source TTC (Kinematic TTC vs Looming TTC) per PRD §24–§28.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from scripts.vehicle_tracker import TrackedObject


class KinematicsEngine:
    """
    Computes inter-vehicle metrics, lead vehicle identification, and TTC analysis.
    """

    def __init__(
        self,
        bus_corridor_half_width: float = 1.8,  # Bus width approx 2.5m, corridor half-width 1.8m
        max_kinematic_range: float = 30.0,
    ):
        self.bus_corridor_half_width = bus_corridor_half_width
        self.max_kinematic_range = max_kinematic_range

    def identify_lead_vehicle(self, tracks: List[TrackedObject]) -> Optional[TrackedObject]:
        """
        Identifies the nearest vehicle directly ahead in the bus travel corridor.
        Per PRD §26: A vehicle behind the bus cannot be a detected lead vehicle or tailgater.
        """
        lead_cand = None
        min_z = float("inf")

        for trk in tracks:
            if trk.category != "vehicle":
                continue
            x, z = trk.current_position_3d
            if z <= 1.0 or z > self.max_kinematic_range:
                continue

            # Check if target is laterally inside the bus corridor
            if abs(x) <= self.bus_corridor_half_width:
                if z < min_z:
                    min_z = z
                    lead_cand = trk

        return lead_cand

    def compute_lead_vehicle_metrics(
        self, lead_trk: Optional[TrackedObject], ego_speed_mps: Optional[float] = None
    ) -> Dict:
        """
        Calculates gap, relative closing rate, time headway, kinematic TTC, and looming TTC.
        """
        if lead_trk is None:
            return {
                "lead_track_id": None,
                "gap_m": None,
                "closing_rate_mps": None,
                "time_headway_s": None,
                "ttc_kinematic_s": None,
                "ttc_looming_s": None,
                "ttc_consensus_s": None,
                "ttc_state": "NO_LEAD_VEHICLE",
            }

        x, z = lead_trk.current_position_3d
        vx, vz = lead_trk.current_velocity_3d

        # Relative closing rate: positive when gap is decreasing (-Vz)
        closing_rate = -vz

        # Time Headway: gap / ego_speed
        # When bus follows lead vehicle, bus is following vehicle
        time_headway = None
        if ego_speed_mps is not None and ego_speed_mps > 1.0 and z > 0:
            time_headway = round(float(z / ego_speed_mps), 2)

        # 1. Kinematic TTC: gap / closing_rate
        ttc_kinematic = None
        if closing_rate > 0.3 and z > 0.5:
            raw_ttc = z / closing_rate
            if 0.1 <= raw_ttc <= 15.0:
                ttc_kinematic = round(float(raw_ttc), 2)

        # 2. Looming TTC: w / dot_w from bounding box width growth
        ttc_looming = self._compute_looming_ttc(lead_trk)

        # 3. Consensus & Uncertainty
        ttc_consensus = None
        ttc_state = "STABLE"

        if ttc_kinematic is not None and ttc_looming is not None:
            if abs(ttc_kinematic - ttc_looming) > 1.8:
                ttc_state = "TTC_UNCERTAIN"
                ttc_consensus = min(ttc_kinematic, ttc_looming)  # Safe conservative estimate
            else:
                ttc_consensus = round(float(0.5 * (ttc_kinematic + ttc_looming)), 2)
                ttc_state = "CONVERGENT"
        elif ttc_kinematic is not None:
            ttc_consensus = ttc_kinematic
            ttc_state = "KINEMATIC_ONLY"
        elif ttc_looming is not None:
            ttc_consensus = ttc_looming
            ttc_state = "LOOMING_ONLY"
        else:
            ttc_state = "NOT_CLOSING"

        # Check range limit
        if z > self.max_kinematic_range:
            ttc_state = "INSUFFICIENT_EVIDENCE_RANGE_TOO_HIGH"
            ttc_consensus = None

        return {
            "lead_track_id": lead_trk.track_id,
            "gap_m": round(float(z), 2),
            "lateral_offset_m": round(float(x), 2),
            "closing_rate_mps": round(float(closing_rate), 2),
            "relative_speed_mps": round(float(vz), 2),
            "time_headway_s": time_headway,
            "ttc_kinematic_s": ttc_kinematic,
            "ttc_looming_s": ttc_looming,
            "ttc_consensus_s": ttc_consensus,
            "ttc_state": ttc_state,
        }

    def _compute_looming_ttc(self, trk: TrackedObject) -> Optional[float]:
        """
        Estimates TTC from apparent bounding box expansion rate:
        TTC_loom = w / (dw/dt)
        """
        if len(trk.history) < 5:
            return None

        # Sample widths over recent 5 frames
        recent = trk.history[-5:]
        widths = [(h["bbox"][2] - h["bbox"][0]) for h in recent]
        timestamps = [h["timestamp"] for h in recent]

        dt_total = timestamps[-1] - timestamps[0]
        if dt_total < 0.2:
            return None

        # Linear regression on width vs time
        t_arr = np.array(timestamps) - timestamps[0]
        w_arr = np.array(widths)

        slope, _ = np.polyfit(t_arr, w_arr, deg=1)  # dw/dt in px/s
        current_w = widths[-1]

        # Only valid if bounding box is expanding significantly (target looming closer)
        if slope > 3.0:  # expanding by at least 3 px/s
            ttc = current_w / slope
            if 0.2 <= ttc <= 15.0:
                return round(float(ttc), 2)

        return None
