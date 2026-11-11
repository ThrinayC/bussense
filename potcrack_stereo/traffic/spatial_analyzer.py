#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Vehicular Perception: Traffic Spatial Analyzer
Derives 3D ground-contact metric position strictly from stereo disparity and road-plane geometry.

Methodology (Fix 2 & Fix 4):
1. Vehicle bbox -> bottom-center ground-contact image point (u_bc, v_bc) = ((x1+x2)/2, y2)
2. Calibrated camera ray -> ray / fitted local road-plane intersection
3. 3D ground-contact position (X, Y, Z) derived from ray-plane intersection
4. Stereo disparity in local ground-contact ROI used as a sanity/validity check
5. Pairwise inter-vehicle distances, longitudinal gap, and ego-to-vehicle distance
6. Same-lane gap labeled explicitly as: 'SAME-LANE HEURISTIC (initial approximation |ΔX| < 1.8m)'
NO LiDAR, NO GT depth, NO monocular depth allowed at deployment inference.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class TrafficSpatialAnalyzer:
    """Calculates metric 3D positions and spatial inter-vehicle distances from stereo disparity."""

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        baseline: float,
        min_valid_disparity: float = 0.5,
        min_depth_m: float = 1.0,
        max_depth_m: float = 80.0,
        same_lane_lateral_thresh_m: float = 1.8
    ):
        """
        Args:
            fx, fy: Camera focal lengths in pixels.
            cx, cy: Principal point offsets in pixels.
            baseline: Stereo baseline in meters.
            min_valid_disparity: Minimum disparity to consider valid.
            min_depth_m, max_depth_m: Plausible depth clipping range in meters.
            same_lane_lateral_thresh_m: Lateral (|delta X|) threshold to consider vehicles in same lane.
        """
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.baseline = float(baseline)
        self.fB = self.fx * self.baseline
        self.min_valid_disparity = min_valid_disparity
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.same_lane_lateral_thresh_m = same_lane_lateral_thresh_m

    def compute_vehicle_3d_position(
        self,
        bbox: List[int],
        disparity: np.ndarray,
        road_plane: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Derive 3D ground-contact position using camera ray / road-plane intersection
        validated against local stereo disparity.

        Steps:
        1. (u_bc, v_bc) = ((x1 + x2)/2, y2)
        2. ray r = [(u_bc - cx)/fx, (v_bc - cy)/fy, 1.0]^T
        3. Local stereo disparity sanity check in ROI near bottom-center
        4. Ray / road-plane intersection: t = -d / (a*rx + b*ry + c) -> (t*rx, t*ry, t)

        Returns:
            Dict containing:
            {
                'valid': bool,
                'X_m': float or None,
                'Y_m': float or None,
                'Z_m': float or None,
                'distance_m': float or None,
                'position_method': str,
                'valid_fraction': float,
                'status': 'OK' | 'DISTANCE: UNRELIABLE'
            }
        """
        H, W = disparity.shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 5 or box_h < 5:
            return {
                "valid": False, "X_m": None, "Y_m": None, "Z_m": None,
                "distance_m": None, "position_method": "none",
                "valid_fraction": 0.0, "status": "DISTANCE: UNRELIABLE"
            }

        # 1. Bottom-center ground-contact image point
        u_bc = (x1 + x2) / 2.0
        v_bc = float(y2)

        # Calibrated camera ray unit direction in camera frame (Z=1 plane)
        rx = (u_bc - self.cx) / self.fx
        ry = (v_bc - self.cy) / self.fy
        rz = 1.0

        # 2. Stereo disparity check in ground-contact ROI: lower 25%, central 60%
        ref_x1 = int(x1 + 0.20 * box_w)
        ref_x2 = int(x2 - 0.20 * box_w)
        ref_y1 = int(y2 - 0.25 * box_h)
        ref_y2 = y2

        ref_region = disparity[ref_y1:ref_y2, ref_x1:ref_x2]
        valid_mask = (ref_region > self.min_valid_disparity) & np.isfinite(ref_region)
        valid_pts = int(np.sum(valid_mask))
        total_pts = int(ref_region.size)

        # Fallback to lower half if central ground-contact has few points
        if valid_pts < 5 and total_pts > 0:
            ref_y1 = int(y1 + 0.50 * box_h)
            ref_region = disparity[ref_y1:ref_y2, x1:x2]
            valid_mask = (ref_region > self.min_valid_disparity) & np.isfinite(ref_region)
            valid_pts = int(np.sum(valid_mask))
            total_pts = int(ref_region.size)

        if valid_pts < 5 or total_pts == 0:
            return {
                "valid": False, "X_m": None, "Y_m": None, "Z_m": None,
                "distance_m": None, "position_method": "insufficient_disparity",
                "valid_fraction": 0.0, "status": "DISTANCE: UNRELIABLE"
            }

        valid_fraction = float(valid_pts / total_pts)
        d_vals = ref_region[valid_mask]

        # Robust disparity estimation using IQR filtering and median
        d_median = float(np.median(d_vals))
        q25, q75 = np.percentile(d_vals, 25), np.percentile(d_vals, 75)
        iqr = q75 - q25
        inliers = d_vals[(d_vals >= q25 - 1.5 * iqr) & (d_vals <= q75 + 1.5 * iqr)]
        d_final = float(np.median(inliers)) if len(inliers) >= 3 else d_median

        if d_final <= self.min_valid_disparity:
            return {
                "valid": False, "X_m": None, "Y_m": None, "Z_m": None,
                "distance_m": None, "position_method": "low_disparity",
                "valid_fraction": round(valid_fraction, 3),
                "status": "DISTANCE: UNRELIABLE"
            }

        # Stereo-derived reference depth
        z_stereo = self.fB / d_final

        # 3. Primary Vehicle Reference: Ray / Fitted Road Plane Intersection
        plane_used = False
        final_X = rx * z_stereo
        final_Y = ry * z_stereo
        final_Z = z_stereo
        method = "stereo_ground_roi"

        if road_plane is not None and len(road_plane) >= 4:
            a, b, c, d = float(road_plane[0]), float(road_plane[1]), float(road_plane[2]), float(road_plane[3])
            denom = a * rx + b * ry + c * rz
            if abs(denom) > 1e-4:
                t_plane = -d / denom
                # Sanity check: road-plane intersection must be positive and consistent with stereo depth
                if self.min_depth_m <= t_plane <= self.max_depth_m:
                    disparity_discrepancy = abs(t_plane - z_stereo)
                    # Allow up to 35% discrepancy or 4.0m between plane ray-cast and stereo disparity
                    if disparity_discrepancy <= max(4.0, 0.35 * z_stereo):
                        final_Z = t_plane
                        final_X = rx * t_plane
                        final_Y = ry * t_plane
                        plane_used = True
                        method = "ray_road_plane_intersection"

        if final_Z < self.min_depth_m or final_Z > self.max_depth_m:
            return {
                "valid": False, "X_m": None, "Y_m": None, "Z_m": None,
                "distance_m": None, "position_method": method,
                "valid_fraction": round(valid_fraction, 3),
                "status": "DISTANCE: UNRELIABLE"
            }

        distance_m = float(np.sqrt(final_X**2 + final_Y**2 + final_Z**2))

        return {
            "valid": True,
            "X_m": round(float(final_X), 2),
            "Y_m": round(float(final_Y), 2),
            "Z_m": round(float(final_Z), 2),
            "distance_m": round(distance_m, 2),
            "position_method": method,
            "valid_fraction": round(valid_fraction, 3),
            "status": "OK"
        }

    def analyze_scene(
        self,
        tracked_vehicles: List[Dict[str, Any]],
        disparity: np.ndarray,
        road_plane: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Localize all tracked vehicles in 3D, compute inter-vehicle distances,
        longitudinal gaps, and ego-to-vehicle metrics.

        Returns:
            Dict containing:
            - 'vehicles': List of vehicle records with 3D positions
            - 'inter_vehicle_distances': List of pairwise distances with SAME-LANE HEURISTIC
            - 'ego_metrics': Telemetry on nearest vehicle and ego longitudinal gap
        """
        localized_vehicles = []
        valid_positions = {}  # display_id -> np.array([X, Y, Z])

        for veh in tracked_vehicles:
            spatial_info = self.compute_vehicle_3d_position(veh["bbox"], disparity, road_plane=road_plane)
            record = dict(veh)
            record.update(spatial_info)
            localized_vehicles.append(record)

            if spatial_info["valid"]:
                disp_id = veh.get("display_id", str(veh.get("track_id")))
                valid_positions[disp_id] = np.array([
                    spatial_info["X_m"], spatial_info["Y_m"], spatial_info["Z_m"]
                ], dtype=np.float32)

        # 1. Pairwise inter-vehicle distances
        inter_vehicle_distances = []
        v_ids = sorted(list(valid_positions.keys()))
        for i in range(len(v_ids)):
            for j in range(i + 1, len(v_ids)):
                id1, id2 = v_ids[i], v_ids[j]
                p1, p2 = valid_positions[id1], valid_positions[id2]
                dist_3d = float(np.linalg.norm(p1 - p2))
                dx = abs(float(p1[0] - p2[0]))
                dz = abs(float(p1[2] - p2[2]))

                # Fix 4: Explicitly labeled SAME-LANE HEURISTIC (initial approximation)
                is_same_lane = bool(dx < self.same_lane_lateral_thresh_m)

                inter_vehicle_distances.append({
                    "vehicle_1": id1,
                    "vehicle_2": id2,
                    "distance_3d_m": round(dist_3d, 2),
                    "longitudinal_gap_m": round(dz, 2) if is_same_lane else None,
                    "same_lane": is_same_lane,
                    "same_lane_rule": "SAME-LANE HEURISTIC (initial approximation |ΔX| < 1.8m, non-universal)"
                })

        # 2. Ego-vehicle proximity
        nearest_id = None
        nearest_class = None
        min_dist = float("inf")
        nearest_gap = None

        for rec in localized_vehicles:
            if rec["valid"] and rec["distance_m"] is not None:
                if rec["distance_m"] < min_dist:
                    min_dist = rec["distance_m"]
                    nearest_id = rec.get("display_id", f"Vehicle #{rec.get('track_id')}")
                    nearest_class = rec.get("class", "vehicle").upper()

                    # Longitudinal gap if ahead
                    if abs(rec["X_m"]) < self.same_lane_lateral_thresh_m:
                        nearest_gap = rec["Z_m"]

        ego_metrics = {
            "nearest_vehicle_id": nearest_id,
            "nearest_vehicle_class": nearest_class,
            "nearest_vehicle_distance_m": round(min_dist, 2) if min_dist != float("inf") else None,
            "nearest_longitudinal_gap_m": round(nearest_gap, 2) if nearest_gap is not None else None,
            "lane_heuristic": "SAME-LANE HEURISTIC (|ΔX| < 1.8m)",
            "status": "OK" if min_dist != float("inf") else "NO_VEHICLES"
        }

        return {
            "vehicles": localized_vehicles,
            "inter_vehicle_distances": inter_vehicle_distances,
            "ego_metrics": ego_metrics
        }
