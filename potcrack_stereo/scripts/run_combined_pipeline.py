#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Phase 12: Unified Combined Perception Pipeline
Runs BOTH branches simultaneously strictly from LEFT RGB + RIGHT RGB:
1. Stereo Disparity & Metric Depth
2. Pothole Segmentation + RANSAC Road Surface Fitting + Shoelace Footprint Metrology
3. Vehicle Detection + ByteTrack Persistent Tracking + 3D Ground-Contact Localization
4. Pairwise Inter-Vehicle Distances + Longitudinal Gap + Ego Proximity
5. Observable Road-Plane Footprint Area & Physical Traffic Density Engine
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic.tracker import TrafficTracker
from traffic.spatial_analyzer import TrafficSpatialAnalyzer
from traffic.density_engine import TrafficDensityEngine
from scripts.stereo_depth import get_stereo_matcher
from scripts.pothole_geometry import PotholeGeometryEngine, PotholeInstance3D


class BusSensePerceptionPipeline:
    """Unified Stereo Perception Pipeline combining Pothole Metrology and Vehicular Tracking."""

    def __init__(
        self,
        calib: Dict[str, float],
        stereo_model: str = "sgbm",
        pothole_weights: Optional[str] = None,
        vehicle_weights: Optional[str] = None,
        conf_pothole: float = 0.20,
        conf_vehicle: float = 0.25,
        device: Optional[str] = None
    ):
        """
        Args:
            calib: Dictionary containing 'fx', 'fy', 'cx', 'cy', 'baseline'
        """
        self.calib = calib
        self.fx = float(calib["fx"])
        self.fy = float(calib["fy"])
        self.cx = float(calib["cx"])
        self.cy = float(calib["cy"])
        self.baseline = float(calib["baseline"])
        self.device = device

        # 1. Stereo Depth Matcher
        self.matcher = get_stereo_matcher(stereo_model)

        # 2. Pothole Segmentation Model
        if pothole_weights is None:
            pothole_weights = str(PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg.pt")
        self.pothole_model = YOLO(pothole_weights)
        self.conf_pothole = conf_pothole

        # 3. Pothole 3D Metrology Engine
        self.pothole_engine = PotholeGeometryEngine(
            min_depth_m=0.5,
            max_depth_m=35.0,
            ring_dilation_px=18,
            min_ring_points=15,
            ransac_threshold_m=0.03
        )

        # 4. Vehicle Tracker (YOLO + ByteTrack)
        if vehicle_weights is None:
            vehicle_weights = "yolov8n.pt"
        self.vehicle_tracker = TrafficTracker(
            model_path=vehicle_weights,
            conf_threshold=conf_vehicle,
            device=device
        )

        # 5. Traffic Spatial Analyzer
        self.spatial_analyzer = TrafficSpatialAnalyzer(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            baseline=self.baseline, min_valid_disparity=0.5
        )

        # 6. Traffic Density Engine
        # Image dimensions will be updated on first frame
        self.density_engine: Optional[TrafficDensityEngine] = None

        # Pothole temporal tracking state
        self.next_pothole_id = 1
        self.active_potholes: Dict[int, Dict[str, Any]] = {}

    def reset(self):
        """Reset temporal state."""
        self.vehicle_tracker.reset()
        if self.density_engine:
            self.density_engine.reset()
        self.next_pothole_id = 1
        self.active_potholes.clear()

    def _estimate_general_road_plane(self, depth_m: np.ndarray) -> Optional[np.ndarray]:
        """
        Fits a general road plane using RANSAC from valid depth points in the lower 35% of the image.
        """
        H, W = depth_m.shape[:2]
        road_roi_y = int(H * 0.65)
        roi_depth = depth_m[road_roi_y:H, :]
        valid = (roi_depth >= 1.0) & (roi_depth <= 35.0) & np.isfinite(roi_depth)
        
        if np.sum(valid) < 50:
            return None

        # Sample up to 1000 points
        ys, xs = np.where(valid)
        if len(ys) > 1000:
            idx = np.random.choice(len(ys), 1000, replace=False)
            ys = ys[idx]
            xs = xs[idx]

        ys_global = ys + road_roi_y
        zs = roi_depth[ys, xs]
        xs_3d = (xs - self.cx) * zs / self.fx
        ys_3d = (ys_global - self.cy) * zs / self.fy

        pts_3d = np.stack([xs_3d, ys_3d, zs], axis=1)

        # Fit plane: Y = m1*X + m2*Z + c  ->  m1*X - Y + m2*Z + c = 0
        from sklearn.linear_model import RANSACRegressor
        X_reg = pts_3d[:, [0, 2]]
        Y_reg = pts_3d[:, 1]
        try:
            ransac = RANSACRegressor(min_samples=10, residual_threshold=0.08, max_trials=100)
            ransac.fit(X_reg, Y_reg)
            inlier_ratio = np.mean(ransac.inlier_mask_)
            if inlier_ratio < 0.40:
                return None
            m1, m2 = ransac.estimator_.coef_
            c_int = ransac.estimator_.intercept_
            # m1*X - 1.0*Y + m2*Z + c_int = 0
            plane = np.array([m1, -1.0, m2, c_int], dtype=np.float32)
            plane /= np.linalg.norm(plane[:3])
            return plane
        except Exception:
            return None

    def _track_potholes(self, current_potholes: List[PotholeInstance3D], iou_thresh: float = 0.25) -> List[Dict[str, Any]]:
        """Associates detected potholes across frames using bounding-box IoU."""
        tracked = []
        assigned_active = set()

        for pot in current_potholes:
            b1 = pot.bbox_xyxy
            best_id = None
            best_iou = 0.0

            for t_id, prev_data in self.active_potholes.items():
                if t_id in assigned_active:
                    continue
                b2 = prev_data["bbox"]
                # Compute IoU
                xi1 = max(b1[0], b2[0])
                yi1 = max(b1[1], b2[1])
                xi2 = min(b1[2], b2[2])
                yi2 = min(b1[3], b2[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter
                iou = inter / max(1.0, union)

                if iou > best_iou:
                    best_iou = iou
                    best_id = t_id

            if best_iou >= iou_thresh and best_id is not None:
                assigned_id = best_id
                assigned_active.add(best_id)
            else:
                assigned_id = self.next_pothole_id
                self.next_pothole_id += 1

            record = {
                "track_id": assigned_id,
                "confidence": pot.confidence,
                "bbox": pot.bbox_xyxy,
                "mask": pot.mask,
                "contour": pot.contour_px,
                "depth_p95_cm": pot.depth_p95_cm,
                "depth_median_cm": pot.depth_median_cm,
                "depth_max_cm": pot.depth_max_cm,
                "area_m2": pot.area_m2,
                "length_m": pot.length_m,
                "width_m": pot.width_m,
                "center_3d": pot.center_3d,
                "plane_equation": pot.plane_equation,
                "quality": pot.quality,
                "quality_reason": pot.quality_reason
            }
            self.active_potholes[assigned_id] = record
            tracked.append(record)

        # Cleanup lost potholes (keep only recently assigned)
        self.active_potholes = {k: v for k, v in self.active_potholes.items() if k in assigned_active or len(tracked) == 0}
        return tracked

    def process_frame(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        frame_index: int = 0
    ) -> Dict[str, Any]:
        """
        Processes one synchronized stereo pair.

        Returns:
            Dict with keys:
            - 'disparity': 2D float32 disparity map
            - 'depth_m': 2D float32 metric depth map
            - 'potholes': List of tracked 3D pothole instances
            - 'traffic': Scene analysis from TrafficSpatialAnalyzer
            - 'density': Telemetry from TrafficDensityEngine
            - 'runtime_ms': Execution timings
        """
        t0 = time.time()
        H, W = left_rgb.shape[:2]

        if self.density_engine is None:
            self.density_engine = TrafficDensityEngine(
                fx=self.fx, img_width=W, img_height=H, z_min=2.0, z_max=35.0
            )

        # Step 1: Stereo Disparity & Depth
        t_stereo0 = time.time()
        disparity = self.matcher.estimate_disparity(left_rgb, right_rgb)
        valid_disp = (disparity > 0.5) & np.isfinite(disparity)
        depth_m = np.zeros_like(disparity, dtype=np.float32)
        depth_m[valid_disp] = (self.fx * self.baseline) / disparity[valid_disp]
        t_stereo = (time.time() - t_stereo0) * 1000.0

        # Step 2: Pothole Segmentation & Metrology Branch
        t_pot0 = time.time()
        pot_res = self.pothole_model(left_rgb, conf=self.conf_pothole, verbose=False)
        calib_dict = {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy, "baseline": self.baseline}
        
        # Extract combined binary mask from YOLO11-seg
        pothole_mask = np.zeros((H, W), dtype=np.uint8)
        if pot_res[0].masks is not None:
            for m in pot_res[0].masks.data.cpu().numpy():
                m_resized = cv2.resize((m > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                pothole_mask = np.bitwise_or(pothole_mask, m_resized * 255)

        pot_instances_raw = self.pothole_engine.analyze_potholes(left_rgb, depth_m, pothole_mask, calib_dict)
        tracked_potholes = self._track_potholes(pot_instances_raw)
        t_pot = (time.time() - t_pot0) * 1000.0

        # Step 3: Road Plane Estimation (from pothole ring or general road surface)
        best_plane = None
        for p in tracked_potholes:
            if p.get("plane_equation") is not None and p["quality"] in ["GOOD", "WARNING"]:
                best_plane = np.array(p["plane_equation"], dtype=np.float32)
                break

        # Fallback: estimate general road plane from lower 35% of stereo depth if no pothole plane
        if best_plane is None:
            best_plane = self._estimate_general_road_plane(depth_m)

        # Step 4: Vehicular Perception Branch (Fix 2: Ray / Road-Plane Intersection)
        t_veh0 = time.time()
        tracked_vehicles = self.vehicle_tracker.track(left_rgb, frame_index=frame_index)
        traffic_scene = self.spatial_analyzer.analyze_scene(
            tracked_vehicles, disparity, road_plane=best_plane
        )
        t_veh = (time.time() - t_veh0) * 1000.0

        # Step 5: Road Footprint Traffic Density (Fix 3: Explicit Road Polygon Area)
        density_telemetry = self.density_engine.update(
            scene_analysis=traffic_scene,
            road_plane=best_plane
        )

        total_time_ms = (time.time() - t0) * 1000.0

        return {
            "frame_index": frame_index,
            "disparity": disparity,
            "depth_m": depth_m,
            "potholes": tracked_potholes,
            "traffic": traffic_scene,
            "density": density_telemetry,
            "runtime_ms": {
                "stereo": round(t_stereo, 1),
                "pothole": round(t_pot, 1),
                "vehicle": round(t_veh, 1),
                "total": round(total_time_ms, 1)
            }
        }
