#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Vehicular Perception: Traffic Density & Occupancy Engine
Estimates physically grounded traffic density over the explicit road-plane polygon.

Methodology (Fix 3):
1. Camera FOV frustum + fitted local road plane (aX + bY + cZ + d = 0)
2. Reliable depth range [Zmin, Zmax] (e.g. 2.0m to 35.0m)
3. Visible lateral road boundaries (clamped to FOV and standard road lane envelope)
4. 4-vertex 3D planar polygon on the road surface
5. Physical polygon area in m² (via 3D vector cross-product / Shoelace metrology)
6. Density = vehicles_in_roi / road_polygon_area (reported in veh/m² and veh/km²)
7. Guardrail: Returns 'DENSITY: UNRELIABLE' if road plane or polygon cannot be estimated reliably.
"""

from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

TRAFFIC_LEVELS = {
    "NORMAL": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "ALARMING": 3
}


class TrafficDensityEngine:
    """Computes traffic density and spatial flow telemetry over the road footprint."""

    def __init__(
        self,
        fx: float,
        img_width: int,
        img_height: int,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        z_min: float = 2.0,
        z_max: float = 35.0,
        max_half_lane_width_m: float = 6.0,
        window_size: int = 60,
        min_inlier_ratio: float = 0.40
    ):
        """
        Args:
            fx: Focal length in pixels.
            img_width, img_height: Camera resolution.
            cx, cy: Principal point offsets (defaults to W/2, H/2 if None).
            z_min, z_max: Observable depth bounds in meters for road evaluation.
            max_half_lane_width_m: Lateral road boundary clamp (meters from center).
            window_size: Sliding window size (frames) for unique vehicle tracking.
            min_inlier_ratio: Minimum RANSAC inlier ratio to trust the road plane.
        """
        self.fx = float(fx)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.cx = float(cx) if cx is not None else float(img_width) / 2.0
        self.cy = float(cy) if cy is not None else float(img_height) / 2.0
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.max_half_lane_width_m = float(max_half_lane_width_m)
        self.window_size = int(window_size)
        self.min_inlier_ratio = float(min_inlier_ratio)

        # Sliding window for tracking unique vehicle IDs over time
        self.history_window = deque(maxlen=self.window_size)
        self.all_time_unique_ids = set()

        # Operational traffic level state & smoothing
        self.TRAFFIC_LEVELS = ["NORMAL", "MEDIUM", "HIGH", "ALARMING"]
        self.traffic_level_history = deque(maxlen=8)
        self.current_traffic_level_idx = 0

    def reset(self):
        """Reset temporal counters."""
        self.history_window.clear()
        self.all_time_unique_ids.clear()
        self.traffic_level_history.clear()
        self.current_traffic_level_idx = 0

    def compute_road_plane_polygon_area(
        self,
        road_plane: Optional[np.ndarray],
        road_plane_inliers: Optional[int] = None,
        total_road_pts: Optional[int] = None
    ) -> Tuple[Optional[float], Optional[np.ndarray], bool, str]:
        """
        Construct explicit 3D polygon on the fitted local road plane and calculate its physical area.

        Camera FOV + Road Plane + [Zmin, Zmax] + Lateral Bounds:
        - Near-Left:  (X_near_left,  Y, Z_min)
        - Near-Right: (X_near_right, Y, Z_min)
        - Far-Right:  (X_far_right,  Y, Z_max)
        - Far-Left:   (X_far_left,   Y, Z_max)

        Returns:
            (polygon_area_m2, polygon_vertices_3d, is_reliable, status_str)
        """
        # Guardrail 1: Check road plane presence
        if road_plane is None or len(road_plane) < 4:
            return None, None, False, "DENSITY: UNRELIABLE (No road plane fitted)"

        a, b, c, d = float(road_plane[0]), float(road_plane[1]), float(road_plane[2]), float(road_plane[3])

        # Guardrail 2: Road plane must have significant vertical normal component (Y axis in camera frame)
        norm = np.sqrt(a * a + b * b + c * c)
        if norm < 1e-6 or abs(b) / norm < 0.20:
            return None, None, False, "DENSITY: UNRELIABLE (Road plane unstable or non-horizontal)"

        # Guardrail 3: RANSAC inlier ratio
        if total_road_pts is not None and total_road_pts > 0:
            inlier_ratio = (road_plane_inliers or 0) / total_road_pts
            if inlier_ratio < self.min_inlier_ratio:
                return None, None, False, f"DENSITY: UNRELIABLE (Road plane inliers {inlier_ratio:.1%} < {self.min_inlier_ratio:.0%})"

        # Compute lateral road boundaries clamped by camera FOV and road envelope
        # At Z_min:
        fov_x_min_near = ((0.0 - self.cx) * self.z_min) / self.fx
        fov_x_max_near = (((self.img_width - 1) - self.cx) * self.z_min) / self.fx
        x_near_left = max(fov_x_min_near, -self.max_half_lane_width_m)
        x_near_right = min(fov_x_max_near, self.max_half_lane_width_m)

        # At Z_max:
        fov_x_min_far = ((0.0 - self.cx) * self.z_max) / self.fx
        fov_x_max_far = (((self.img_width - 1) - self.cx) * self.z_max) / self.fx
        x_far_left = max(fov_x_min_far, -self.max_half_lane_width_m)
        x_far_right = min(fov_x_max_far, self.max_half_lane_width_m)

        # Solve for Y on plane: Y = -(a*X + c*Z + d) / b
        def plane_y(x_val: float, z_val: float) -> float:
            return -(a * x_val + c * z_val + d) / b

        v1 = np.array([x_near_left, plane_y(x_near_left, self.z_min), self.z_min], dtype=np.float64)
        v2 = np.array([x_near_right, plane_y(x_near_right, self.z_min), self.z_min], dtype=np.float64)
        v3 = np.array([x_far_right, plane_y(x_far_right, self.z_max), self.z_max], dtype=np.float64)
        v4 = np.array([x_far_left, plane_y(x_far_left, self.z_max), self.z_max], dtype=np.float64)
        vertices = np.vstack([v1, v2, v3, v4])

        # Compute 3D polygon area using vector cross products:
        # A_vec = 0.5 * sum(v_i x v_{i+1})
        cross_sum = (
            np.cross(v1, v2) +
            np.cross(v2, v3) +
            np.cross(v3, v4) +
            np.cross(v4, v1)
        )
        polygon_area = 0.5 * float(np.linalg.norm(cross_sum))

        # Guardrail 4: Plausible road surface area
        if polygon_area < 10.0 or not np.isfinite(polygon_area):
            return None, vertices, False, f"DENSITY: UNRELIABLE (Area {polygon_area:.1f}m² < 10m²)"

        return round(polygon_area, 2), vertices, True, "OK"

    def update(
        self,
        scene_analysis: Dict[str, Any],
        road_plane: Optional[np.ndarray] = None,
        road_plane_inliers: Optional[int] = None,
        total_road_pts: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate full traffic density and flow metrics for current frame.

        Args:
            scene_analysis: Output from TrafficSpatialAnalyzer.analyze_scene()
            road_plane: Optional fitted road plane [a, b, c, d]
            road_plane_inliers: Count of RANSAC inliers on road surface
            total_road_pts: Count of candidate road sample points

        Returns:
            Dict of traffic telemetry.
        """
        vehicles = scene_analysis.get("vehicles", [])
        inter_dists = scene_analysis.get("inter_vehicle_distances", [])

        visible_count = len(vehicles)
        tracked_count = sum(1 for v in vehicles if v.get("track_id", -1) >= 0)

        # Count vehicles positioned within the observable road ROI [z_min, z_max]
        vehicles_in_roi = 0
        current_frame_ids = set()

        for v in vehicles:
            t_id = v.get("track_id", -1)
            if t_id >= 0:
                current_frame_ids.add(t_id)
                self.all_time_unique_ids.add(t_id)

            if v.get("valid", False) and v.get("Z_m") is not None:
                z = v["Z_m"]
                if self.z_min <= z <= self.z_max:
                    vehicles_in_roi += 1
            else:
                # If distance unmeasured, include if in road region
                y2 = v["bbox"][3]
                if y2 > 0.20 * self.img_height:
                    vehicles_in_roi += 1

        # Update temporal sliding window
        self.history_window.append(current_frame_ids)
        window_unique_ids = set().union(*self.history_window) if self.history_window else set()

        # Fix 3: Explicit road-plane polygon area calculation
        area_m2, vertices_3d, is_reliable, area_status = self.compute_road_plane_polygon_area(
            road_plane, road_plane_inliers, total_road_pts
        )

        if is_reliable and area_m2 is not None and area_m2 > 0:
            density_veh_m2 = vehicles_in_roi / area_m2
            density_veh_km2 = density_veh_m2 * 1_000_000.0
            density_status = "OK"
        else:
            density_veh_m2 = None
            density_veh_km2 = None
            density_status = "DENSITY: UNRELIABLE"

        # Inter-vehicle spacing statistics
        valid_3d_dists = [d["distance_3d_m"] for d in inter_dists if d.get("distance_3d_m") is not None]
        same_lane_gaps = [d["longitudinal_gap_m"] for d in inter_dists if d.get("longitudinal_gap_m") is not None]

        avg_spacing = float(np.mean(valid_3d_dists)) if valid_3d_dists else None
        median_spacing = float(np.median(valid_3d_dists)) if valid_3d_dists else None
        nearest_gap = float(np.min(same_lane_gaps)) if same_lane_gaps else (
            float(np.min(valid_3d_dists)) if valid_3d_dists else None
        )

        # Unified Traffic Evidence Model (Approved Specification)
        # Normalized sub-evidence components [0.0, 1.0]:
        e_count = min(1.0, float(tracked_count) / 6.0)
        e_density = min(1.0, float(density_veh_km2) / 12000.0) if (density_veh_km2 is not None and density_veh_km2 > 0) else 0.0
        e_headway = max(0.0, 1.0 - float(nearest_gap) / 20.0) if (nearest_gap is not None and nearest_gap >= 0) else 0.0
        e_occupancy = min(1.0, float(vehicles_in_roi) / 5.0)

        # Weighted composite score:
        # Weights: count=0.30, density=0.25, headway=0.30, occupancy=0.15
        traffic_score = round(0.30 * e_count + 0.25 * e_density + 0.30 * e_headway + 0.15 * e_occupancy, 4)

        score_breakdown = {
            "e_count": round(e_count, 3),
            "e_density": round(e_density, 3),
            "e_headway": round(e_headway, 3),
            "e_occupancy": round(e_occupancy, 3),
            "composite_score": traffic_score
        }

        # Ordered precedence classification:
        # ALARMING: traffic_score >= 0.75 OR critical headway (>= 3 vehicles with headway <= 4.0m)
        # HIGH:     traffic_score >= 0.50
        # MEDIUM:   traffic_score >= 0.25
        # NORMAL:   traffic_score < 0.25
        if traffic_score >= 0.75 or (tracked_count >= 3 and nearest_gap is not None and nearest_gap <= 4.0):
            instant_traffic_level = 3  # ALARMING
        elif traffic_score >= 0.50:
            instant_traffic_level = 2  # HIGH
        elif traffic_score >= 0.25:
            instant_traffic_level = 1  # MEDIUM
        else:
            instant_traffic_level = 0  # NORMAL

        self.traffic_level_history.append(instant_traffic_level)

        # Asymmetric temporal hysteresis: 2 frames up, 4 frames down
        if len(self.traffic_level_history) >= 2:
            last_two = list(self.traffic_level_history)[-2:]
            if all(lvl > self.current_traffic_level_idx for lvl in last_two):
                self.current_traffic_level_idx = min(last_two)
            elif len(self.traffic_level_history) >= 4 and all(lvl < self.current_traffic_level_idx for lvl in list(self.traffic_level_history)[-4:]):
                self.current_traffic_level_idx = max(list(self.traffic_level_history)[-4:])
        else:
            self.current_traffic_level_idx = instant_traffic_level

        final_traffic_level = self.TRAFFIC_LEVELS[self.current_traffic_level_idx]

        return {
            "visible_vehicle_count": visible_count,
            "tracked_vehicle_count": tracked_count,
            "vehicles_in_roi": vehicles_in_roi,
            "unique_vehicle_count_window": len(window_unique_ids),
            "total_unique_vehicles": len(self.all_time_unique_ids),
            "observable_road_area_m2": area_m2,
            "traffic_density_veh_m2": round(density_veh_m2, 5) if density_veh_m2 is not None else None,
            "traffic_density_veh_km2": round(density_veh_km2, 1) if density_veh_km2 is not None else None,
            "density_status": density_status,
            "average_spacing_m": round(avg_spacing, 1) if avg_spacing is not None else None,
            "median_spacing_m": round(median_spacing, 1) if median_spacing is not None else None,
            "nearest_gap_m": round(nearest_gap, 1) if nearest_gap is not None else None,
            "traffic_score": traffic_score,
            "traffic_score_breakdown": score_breakdown,
            "traffic_level": final_traffic_level,
            "traffic_level_raw": self.TRAFFIC_LEVELS[instant_traffic_level]
        }
