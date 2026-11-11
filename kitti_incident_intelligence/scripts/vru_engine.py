"""
Vulnerable Road User (VRU) Intelligence Engine — Phase 5
Evaluates Pedestrian-in-path, Pedestrian crossing, Cyclist conflict, VRU near-miss,
and Small-stature pedestrian candidates per PRD §14–§20.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from scripts.vehicle_tracker import TrackedObject


class VRUEngine:
    """
    Evaluates spatial corridor intersections and interactions with Vulnerable Road Users.
    """

    def __init__(
        self,
        calib,
        corridor_half_width: float = 1.8,
        corridor_max_range: float = 25.0,
        time_to_path_horizon: float = 3.0,
    ):
        self.calib = calib
        self.corridor_half_width = corridor_half_width
        self.corridor_max_range = corridor_max_range
        self.time_to_path_horizon = time_to_path_horizon

    def _associate_riders(self, tracks: List[TrackedObject]):
        """
        Associates 'person' detections that are riding bicycles or motorcycles.
        Suppresses false PEDESTRIAN_IN_PATH alerts when the person is part of a cyclist/rider entity.
        """
        # Reset previous associations
        for trk in tracks:
            trk.is_associated_rider = False
            trk.associated_bike_id = None

        bike_tracks = [t for t in tracks if getattr(t, "stable_class", t.cls_name) in ["bicycle", "motorcycle"]]
        person_tracks = [t for t in tracks if getattr(t, "stable_class", t.cls_name) == "person"]

        for person in person_tracks:
            px, pz = person.current_position_3d
            pvx, pvz = person.current_velocity_3d
            pb = person.bbox

            for bike in bike_tracks:
                bx, bz = bike.current_position_3d
                bvx, bvz = bike.current_velocity_3d
                bb = bike.bbox

                # 3D spatial separation
                dz = abs(pz - bz)
                dx = abs(px - bx)
                dvz = abs(pvz - bvz)

                # 2D Bbox overlap / vertical proximity
                x_overlap = max(0, min(pb[2], bb[2]) - max(pb[0], bb[0]))
                y_overlap = max(0, min(pb[3], bb[3]) - max(pb[1], bb[1]))
                has_2d_overlap = (x_overlap > 0 and y_overlap > 0)

                pb_cx = 0.5 * (pb[0] + pb[2])
                bb_cx = 0.5 * (bb[0] + bb[2])
                cx_dist_px = abs(pb_cx - bb_cx)
                bike_w = bb[2] - bb[0]

                # Match condition:
                if (dz < 2.5 and dx < 1.5 and dvz < 2.5) and (has_2d_overlap or cx_dist_px < bike_w * 1.2):
                    person.is_associated_rider = True
                    person.associated_bike_id = bike.track_id
                    break

    def evaluate_vru_tracks(
        self, tracks: List[TrackedObject], ego_speed_mps: float = 0.0
    ) -> List[Dict]:
        """
        Evaluates all active VRU tracks for path intersections, crossing behavior,
        conflicts, near-misses, and small-stature candidacy.
        Disambiguates cyclists and pedestrians, associating mounted riders.
        """
        self._associate_riders(tracks)
        vru_events: List[Dict] = []

        for trk in tracks:
            if trk.category != "vru":
                continue
            if trk.hits < 3:
                # Require multiple temporal observations per PRD §17
                continue

            x, z = trk.current_position_3d
            vx, vz = trk.current_velocity_3d
            eff_class = getattr(trk, "stable_class", trk.cls_name)
            is_rider = getattr(trk, "is_associated_rider", False)

            # Range gating: beyond 25m geometry is insufficient for VRU conflict
            if z > self.corridor_max_range or z < 0.5:
                continue

            # 1. Check Bus Corridor Intersection
            in_corridor_now = abs(x) <= self.corridor_half_width
            
            # Predict future lateral position: X(t) = X + Vx * t
            time_to_corridor = None
            path_intersects = False

            if in_corridor_now:
                path_intersects = True
                time_to_corridor = 0.0
            elif abs(vx) > 0.2:
                # Time to reach corridor boundary
                target_boundary = self.corridor_half_width if x < 0 else -self.corridor_half_width
                dist_to_boundary = abs(x) - self.corridor_half_width
                if (x < 0 and vx > 0) or (x > 0 and vx < 0):
                    t_reach = dist_to_boundary / abs(vx)
                    if t_reach <= self.time_to_path_horizon:
                        path_intersects = True
                        time_to_corridor = round(float(t_reach), 2)

            # 2. Crossing Evaluation (sustained lateral motion across corridor)
            is_crossing = False
            if len(trk.history) >= 4 and abs(vx) > 0.4:
                # Verify direction of motion has persisted
                recent_vxs = [h["Vx"] for h in trk.history[-4:]]
                if all(v > 0.2 for v in recent_vxs) or all(v < -0.2 for v in recent_vxs):
                    is_crossing = True

            # 3. Time to Collision (TTC) for VRU
            closing_rate = -vz
            vru_ttc = None
            if closing_rate > 0.3 and z > 0.5:
                raw_ttc = z / closing_rate
                if 0.1 <= raw_ttc <= 8.0:
                    vru_ttc = round(float(raw_ttc), 2)

            # 4. Small-Stature Pedestrian Candidate (PRD §15 & §60)
            if is_rider:
                # Discard small-stature candidacy for mounted riders due to bent posture
                small_stature_result = {"is_candidate": False, "estimated_height_m": None, "height_uncertainty_m": None, "reliability": "DISCARDED_MOUNTED_RIDER"}
            else:
                small_stature_result = self._evaluate_small_stature(trk)

            # 5. Event Classification with Semantic Specificity Priority
            # CYCLIST > PEDESTRIAN > MOTORCYCLE
            events_for_track = []

            # A. In Path Alerts
            if path_intersects:
                if eff_class == "bicycle":
                    events_for_track.append({
                        "event_type": "CYCLIST_IN_PATH",
                        "time_to_path_s": time_to_corridor,
                    })
                elif eff_class == "motorcycle":
                    events_for_track.append({
                        "event_type": "MOTORCYCLE_IN_PATH",
                        "time_to_path_s": time_to_corridor,
                    })
                elif eff_class == "person":
                    if is_rider:
                        # RIDER SUPPRESSION: Rider is associated with bike.
                        # SUPPRESS PEDESTRIAN_IN_PATH, emit CYCLIST_IN_PATH
                        events_for_track.append({
                            "event_type": "CYCLIST_IN_PATH",
                            "time_to_path_s": time_to_corridor,
                            "composite_rider": True,
                        })
                    else:
                        events_for_track.append({
                            "event_type": "PEDESTRIAN_IN_PATH",
                            "time_to_path_s": time_to_corridor,
                        })

            # B. Crossing Alerts
            if is_crossing:
                if eff_class == "bicycle" or is_rider:
                    events_for_track.append({
                        "event_type": "CYCLIST_CROSSING",
                        "lateral_speed_mps": round(float(abs(vx)), 2),
                    })
                elif eff_class == "person":
                    events_for_track.append({
                        "event_type": "PEDESTRIAN_CROSSING",
                        "lateral_speed_mps": round(float(abs(vx)), 2),
                    })

            # C. VRU Near-Miss Candidate (PRD §20)
            is_near_miss = False
            if in_corridor_now and z <= 8.0 and (vru_ttc is not None and vru_ttc < 2.5):
                is_near_miss = True
                events_for_track.append({
                    "event_type": "VRU_NEAR_MISS",
                    "near_miss_distance_m": round(float(z), 2),
                    "near_miss_ttc_s": vru_ttc,
                })

            # Display class: if rider, present as cyclist
            display_vru_class = "cyclist" if (eff_class == "bicycle" or is_rider) else eff_class

            # Build comprehensive VRU observation
            vru_obs = {
                "track_id": trk.track_id,
                "vru_class": display_vru_class,
                "is_associated_rider": is_rider,
                "associated_bike_id": getattr(trk, "associated_bike_id", None),
                "distance_m": round(float(z), 2),
                "lateral_offset_m": round(float(x), 2),
                "relative_speed_mps": round(float(vz), 2),
                "lateral_velocity_mps": round(float(vx), 2),
                "in_path": path_intersects,
                "in_corridor_now": in_corridor_now,
                "crossing": is_crossing,
                "near_miss": is_near_miss,
                "ttc_s": vru_ttc,
                "small_stature_candidate": small_stature_result["is_candidate"],
                "estimated_height_m": small_stature_result["estimated_height_m"],
                "height_uncertainty_m": small_stature_result["height_uncertainty_m"],
                "height_reliability": small_stature_result["reliability"],
                "events": events_for_track,
                "state": "UNVERIFIED",
            }
            vru_events.append(vru_obs)

        return vru_events

    def _evaluate_small_stature(self, trk: TrackedObject) -> Dict:
        """
        Estimates observable pedestrian physical height:
        H ≈ h_px * Z / fy
        Classifies as SMALL_STATURE_PEDESTRIAN_CANDIDATE (UNCERTAIN) per PRD §15.
        Never outputs child detection.
        """
        if trk.cls_name != "person":
            return {"is_candidate": False, "estimated_height_m": None, "height_uncertainty_m": None, "reliability": "NOT_PERSON"}

        x1, y1, x2, y2 = trk.bbox
        h_px = y2 - y1
        _, z = trk.current_position_3d

        # Check truncation: if bounding box touches image boundaries
        is_truncated = (
            y1 <= 2 or y2 >= (self.calib.image_height - 2) or x1 <= 2 or x2 >= (self.calib.image_width - 2)
        )

        if is_truncated or z > 20.0 or z < 1.5:
            return {
                "is_candidate": False,
                "estimated_height_m": None,
                "height_uncertainty_m": None,
                "reliability": "UNRELIABLE_TRUNCATED_OR_FAR",
            }

        # Height estimate:
        est_height = float(h_px * z / self.calib.fy)

        # Height uncertainty: delta_H = (delta_h * Z / f) + (h * delta_Z / f)
        delta_h = 4.0  # px bounding box jitter
        delta_z = trk.kf.get_state()["sigma_Z"]
        sigma_h = (delta_h * z / self.calib.fy) + (h_px * delta_z / self.calib.fy)

        # Small stature threshold: typical adult height > 1.5m. Candidate if height < 1.35m
        is_small = (est_height < 1.35) and (est_height > 0.6)

        return {
            "is_candidate": is_small,
            "estimated_height_m": round(est_height, 2),
            "height_uncertainty_m": round(float(sigma_h), 2),
            "reliability": "OBSERVABLE_PERSPECTIVE_ESTIMATE",
        }
