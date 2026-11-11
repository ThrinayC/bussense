"""
Incident Intelligence Engine — Phase 7
Deterministic, evidence-based kinematic state machines for traffic events per PRD §30–§38.
Generates candidate events conforming strictly to the central event schema (§47).
All events are emitted with status UNVERIFIED.
"""

from typing import Dict, List, Optional
import numpy as np

from scripts.vehicle_tracker import TrackedObject
from scripts.ego_motion import EgoMotionFrame


class IncidentEngine:
    """
    Evaluates traffic kinematics against deterministic threshold state machines.
    """

    def __init__(
        self,
        max_validated_range_m: float = 25.0,
        rapid_closing_rate_mps: float = 3.5,
        rapid_closing_max_ttc_s: float = 5.5,
        dangerous_following_max_headway_s: float = 0.8,
        hard_braking_accel_mps2: float = -3.5,
        cut_in_lateral_speed_mps: float = 0.8,
    ):
        self.max_validated_range_m = max_validated_range_m
        self.rapid_closing_rate_mps = rapid_closing_rate_mps
        self.rapid_closing_max_ttc_s = rapid_closing_max_ttc_s
        self.dangerous_following_max_headway_s = dangerous_following_max_headway_s
        self.hard_braking_accel_mps2 = hard_braking_accel_mps2
        self.cut_in_lateral_speed_mps = cut_in_lateral_speed_mps

        self.event_counter = 0
        self.persistence: Dict[str, int] = {}

    def evaluate_traffic_events(
        self,
        tracks: List[TrackedObject],
        lead_metrics: Dict,
        ego_frame: Optional[EgoMotionFrame],
        frame_id: int,
        timestamp: float,
    ) -> List[Dict]:
        """
        Evaluates all active vehicle tracks and lead vehicle metrics for incident events.
        Enforces temporal persistence confirmation to suppress single-frame derivative noise.
        """
        events: List[Dict] = []
        active_conditions = set()

        # 1. Evaluate Lead Vehicle Events
        lead_trk_id = lead_metrics.get("lead_track_id")
        gap = lead_metrics.get("gap_m")
        closing_rate = lead_metrics.get("closing_rate_mps")
        headway = lead_metrics.get("time_headway_s")
        ttc = lead_metrics.get("ttc_consensus_s")

        if lead_trk_id is not None and gap is not None:
            # Range gating check
            if gap <= self.max_validated_range_m:
                # A. RAPID_CLOSING
                rc_key = f"lead_{lead_trk_id}_RAPID_CLOSING"
                rc_cond = (
                    closing_rate is not None
                    and closing_rate >= self.rapid_closing_rate_mps
                    and (ttc is not None and ttc <= self.rapid_closing_max_ttc_s)
                )
                if self._update_persistence(rc_key, rc_cond, min_frames=3):
                    active_conditions.add(rc_key)
                    ev = self._create_event_package(
                        event_type="RAPID_CLOSING",
                        track_id=lead_trk_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        distance_m=gap,
                        closing_rate_mps=closing_rate,
                        time_headway_s=headway,
                        ttc_s=ttc,
                        ego_frame=ego_frame,
                    )
                    events.append(ev)

                # B. DANGEROUS_FOLLOWING (Bus following lead vehicle too closely per PRD §26 & §32)
                df_key = f"lead_{lead_trk_id}_DANGEROUS_FOLLOWING"
                df_cond = (
                    headway is not None
                    and headway < self.dangerous_following_max_headway_s
                    and ego_frame is not None
                    and ego_frame.speed_mps > 4.0
                )
                if self._update_persistence(df_key, df_cond, min_frames=3):
                    active_conditions.add(df_key)
                    ev = self._create_event_package(
                        event_type="DANGEROUS_FOLLOWING",
                        track_id=lead_trk_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        distance_m=gap,
                        closing_rate_mps=closing_rate or 0.0,
                        time_headway_s=headway,
                        ttc_s=ttc,
                        ego_frame=ego_frame,
                    )
                    events.append(ev)

        # 2. Evaluate Per-Track Kinematics (Hard Braking, Cut-In, Weaving, Collision Candidate)
        for trk in tracks:
            # Only evaluate fully active tracks with confirmed history
            if trk.category != "vehicle" or trk.time_since_update > 0 or len(trk.history) < 4:
                continue

            x, z = trk.current_position_3d
            vx, vz = trk.current_velocity_3d

            if z > self.max_validated_range_m or z < 1.0:
                continue

            # C. HARD_BRAKING (sustained negative longitudinal acceleration)
            hb_key = f"{trk.track_id}_HARD_BRAKING"
            accel_z = self._estimate_acceleration(trk)
            hb_cond = accel_z < self.hard_braking_accel_mps2
            if self._update_persistence(hb_key, hb_cond, min_frames=3):
                active_conditions.add(hb_key)
                ev = self._create_event_package(
                    event_type="HARD_BRAKING",
                    track_id=trk.track_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    distance_m=round(float(z), 2),
                    closing_rate_mps=round(float(-vz), 2),
                    ttc_s=round(float(z / max(0.1, -vz)), 2) if -vz > 0.5 else None,
                    ego_frame=ego_frame,
                    extra_metrics={"target_accel_mps2": round(float(accel_z), 2)},
                )
                events.append(ev)

            # D. CUT_IN_CANDIDATE (Lateral displacement entering forward corridor)
            ci_key = f"{trk.track_id}_CUT_IN_CANDIDATE"
            ci_cond = self._is_cut_in(trk)
            if self._update_persistence(ci_key, ci_cond, min_frames=2):
                active_conditions.add(ci_key)
                ev = self._create_event_package(
                    event_type="CUT_IN_CANDIDATE",
                    track_id=trk.track_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    distance_m=round(float(z), 2),
                    closing_rate_mps=round(float(-vz), 2),
                    ttc_s=None,
                    ego_frame=ego_frame,
                    extra_metrics={"lateral_velocity_mps": round(float(vx), 2)},
                )
                events.append(ev)

            # E. WEAVING_CANDIDATE (Repeated lateral direction reversals)
            wv_key = f"{trk.track_id}_WEAVING_CANDIDATE"
            wv_cond = self._is_weaving(trk)
            if self._update_persistence(wv_key, wv_cond, min_frames=1):
                active_conditions.add(wv_key)
                ev = self._create_event_package(
                    event_type="WEAVING_CANDIDATE",
                    track_id=trk.track_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    distance_m=round(float(z), 2),
                    closing_rate_mps=round(float(-vz), 2),
                    ttc_s=None,
                    ego_frame=ego_frame,
                )
                events.append(ev)

            # F. COLLISION_CANDIDATE (PRD §37 & §38)
            # Imminent collision risk: low gap + high closing rate, or critical TTC with high closing rate
            cc_key = f"{trk.track_id}_COLLISION_CANDIDATE"
            ttc_est = (z / max(0.1, -vz)) if -vz > 0.5 else 999.0
            cc_cond = (z <= 4.0 and -vz > 2.5) or (ttc_est <= 2.2 and -vz >= 4.0 and z <= 15.0)
            if self._update_persistence(cc_key, cc_cond, min_frames=2):
                active_conditions.add(cc_key)
                ev = self._create_event_package(
                    event_type="COLLISION_CANDIDATE",
                    track_id=trk.track_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    distance_m=round(float(z), 2),
                    closing_rate_mps=round(float(-vz), 2),
                    ttc_s=round(float(ttc_est), 2),
                    ego_frame=ego_frame,
                )
                events.append(ev)

        # Clear expired persistence keys
        stale_keys = [k for k in self.persistence if k not in active_conditions and self.persistence[k] == 0]
        for k in stale_keys:
            del self.persistence[k]

        return events

    def _update_persistence(self, key: str, condition_met: bool, min_frames: int) -> bool:
        """Updates frame persistence counter and returns True if sustained for min_frames."""
        if condition_met:
            self.persistence[key] = self.persistence.get(key, 0) + 1
            return self.persistence[key] >= min_frames
        else:
            self.persistence[key] = 0
            return False

    def _estimate_acceleration(self, trk: TrackedObject) -> float:
        """Estimates d(Vz)/dt via robust linear regression over the last 5 frames."""
        if len(trk.history) < 4:
            return 0.0
        v_list = [h["Vz"] for h in trk.history[-5:]]
        t_list = [h["timestamp"] for h in trk.history[-5:]]
        dt = t_list[-1] - t_list[0]
        if dt < 0.2:
            return 0.0
        t_arr = np.array(t_list) - t_list[0]
        v_arr = np.array(v_list)
        denom = np.sum((t_arr - np.mean(t_arr)) ** 2)
        if denom < 1e-4:
            return 0.0
        return float(np.sum((t_arr - np.mean(t_arr)) * (v_arr - np.mean(v_arr))) / denom)

    def _is_cut_in(self, trk: TrackedObject) -> bool:
        """Verifies lateral trajectory crossing from outside into the bus forward corridor."""
        if len(trk.history) < 4:
            return False
        old_x = trk.history[-4]["X"]
        current_x = trk.history[-1]["X"]
        current_z = trk.history[-1]["Z"]
        vx = trk.history[-1]["Vx"]

        # Vehicle must have originated outside the corridor (|X| > 1.7m)
        was_outside = abs(old_x) > 1.7
        # Now entering or inside the corridor
        now_inside = abs(current_x) <= 1.7
        # Lateral velocity directed inward toward the corridor center
        moving_inward = (old_x > 1.7 and vx < -0.35) or (old_x < -1.7 and vx > 0.35)

        return was_outside and now_inside and moving_inward and (current_z <= self.max_validated_range_m)

    def _is_weaving(self, trk: TrackedObject) -> bool:
        """Verifies repeated lateral velocity reversals within recent history."""
        if len(trk.history) < 6:
            return False
        vxs = [h["Vx"] for h in trk.history[-8:]]
        signs = [1 if v > 0.3 else (-1 if v < -0.3 else 0) for v in vxs]
        compact = []
        for s in signs:
            if s != 0 and (not compact or compact[-1] != s):
                compact.append(s)
        return len(compact) >= 3

    def _create_event_package(
        self,
        event_type: str,
        track_id: int,
        frame_id: int,
        timestamp: float,
        distance_m: float,
        closing_rate_mps: float,
        time_headway_s: Optional[float] = None,
        ttc_s: Optional[float] = None,
        ego_frame: Optional[EgoMotionFrame] = None,
        extra_metrics: Optional[Dict] = None,
    ) -> Dict:
        """Conforms strictly to the central event schema in PRD §47."""
        self.event_counter += 1
        event_id = f"EVT_{frame_id:04d}_{self.event_counter:04d}_{event_type}"

        schema = {
            "event_id": event_id,
            "event_type": event_type,
            "state": "UNVERIFIED",
            "timestamp": timestamp,
            "frame_id": frame_id,
            "gps": {
                "lat": ego_frame.lat if ego_frame else None,
                "lon": ego_frame.lon if ego_frame else None,
            },
            "ego": {
                "speed_mps": ego_frame.speed_mps if ego_frame else None,
                "yaw_rate": ego_frame.w_yaw if ego_frame else None,
            },
            "vehicles": [{"track_id": track_id}],
            "vru": {"track_id": None, "class": None},
            "metrics": {
                "distance_m": distance_m,
                "relative_speed_mps": round(-closing_rate_mps, 2),
                "closing_rate_mps": closing_rate_mps,
                "time_headway_s": time_headway_s,
                "ttc_kinematic_s": ttc_s,
                "ttc_looming_s": None,
            },
            "uncertainty": {
                "evidence_range_m": distance_m,
                "max_supported_range_m": self.max_validated_range_m,
                "range_qualified": distance_m <= self.max_validated_range_m,
            },
            "registration": {"number": None, "confidence": None},
            "evidence": {
                "pre_event": [],
                "event": [frame_id],
                "post_event": [],
                "clip": None,
                "vehicle_crop": None,
                "plate_crop": None,
            },
        }

        if extra_metrics:
            schema["metrics"].update(extra_metrics)

        return schema
