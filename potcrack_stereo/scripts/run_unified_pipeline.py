#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Unified Perception Pipeline
======================================
Fuses all working BusSense perception modules into ONE unified stereo pipeline:
- Stereo depth (SGBM / Fast-FoundationStereo) computed ONCE per stereo pair
- Pothole segmentation (YOLO11-seg) + local road-plane fitting + 3D depth & footprint area
- Vehicle detection (YOLO) + ByteTrack multi-object tracking
- 3D ground-contact vehicle localization (calibrated ray / road-plane intersection)
- Pairwise inter-vehicle distances + SAME-LANE HEURISTIC longitudinal gap
- Ego proximity tracking
- Observable road-surface 3D polygon area & physical traffic density engine
- Comprehensive UnifiedFrameState object

Strict deployment rule:
Input: LEFT RGB + RIGHT RGB only (Zero LiDAR, Zero monocular depth, Zero GT at inference).
"""

import os
import sys
import time
from dataclasses import dataclass, field, asdict
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
from scripts.crack_geometry import CrackGeometryEngine, CrackInstance3D
from scripts.road_quality import RoadQualityEngine, RoadQualityMetrics


@dataclass
class UnifiedFrameState:
    """Complete unified state for a single stereo perception frame."""
    frame_id: int
    timestamp: float
    vehicles: List[Dict[str, Any]]
    potholes: List[Dict[str, Any]]
    cracks: List[Dict[str, Any]]
    road_quality: Dict[str, Any]
    traffic_metrics: Dict[str, Any]
    road_plane: Dict[str, Any]
    stereo_quality: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    disparity: Optional[np.ndarray] = field(default=None, repr=False)
    depth_m: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Serializes state to dict, optionally omitting heavy numpy arrays."""
        d = {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 4),
            "vehicles": self.vehicles,
            "potholes": self.potholes,
            "cracks": self.cracks,
            "road_quality": self.road_quality,
            "traffic_metrics": self.traffic_metrics,
            "road_plane": self.road_plane,
            "stereo_quality": self.stereo_quality,
            "performance_metrics": self.performance_metrics,
        }
        if include_arrays:
            d["disparity"] = self.disparity
            d["depth_m"] = self.depth_m
        return d

    def __getitem__(self, key: str) -> Any:
        """Enables backward-compatible dictionary access for existing scripts."""
        if key == "frame_index" or key == "frame_id":
            return self.frame_id
        if key == "vehicles":
            return self.vehicles
        if key == "potholes":
            return self.potholes
        if key == "cracks":
            return self.cracks
        if key == "road_quality":
            return self.road_quality
        if key == "traffic":
            return {
                "vehicles": self.vehicles,
                "ego_metrics": {
                    "nearest_vehicle_distance_m": self.traffic_metrics.get("nearest_vehicle_distance_m"),
                    "nearest_vehicle_class": self.traffic_metrics.get("nearest_vehicle_class"),
                    "nearest_vehicle_id": self.traffic_metrics.get("nearest_vehicle_id"),
                    "nearest_longitudinal_gap_m": self.traffic_metrics.get("nearest_longitudinal_gap_m"),
                    "average_longitudinal_gap_m": self.traffic_metrics.get("average_longitudinal_gap_m"),
                },
                "pairwise_distances": self.traffic_metrics.get("pairwise_distances", [])
            }
        if key == "density":
            return self.traffic_metrics
        if key == "traffic_metrics":
            return self.traffic_metrics
        if key == "road_plane":
            return self.road_plane
        if key == "stereo_quality":
            return self.stereo_quality
        if key == "performance_metrics" or key == "runtime_ms":
            return self.performance_metrics
        if key == "disparity":
            return self.disparity
        if key == "depth_m":
            return self.depth_m
        raise KeyError(f"UnifiedFrameState has no attribute/key '{key}'")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except KeyError:
            return False


class UnifiedBusSensePipeline:
    """
    Unified BusSense Perception Engine:
    Single-pass stereo depth feeding parallel pothole metrology and traffic branches.
    """

    def __init__(
        self,
        calib: Dict[str, float],
        stereo_model: str = "sgbm",
        pothole_weights: Optional[str] = None,
        vehicle_weights: Optional[str] = None,
        crack_weights: Optional[str] = None,
        conf_pothole: float = 0.20,
        conf_vehicle: float = 0.20,
        conf_crack: float = 0.035,
        device: Optional[str] = None,
        random_seed: int = 42
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
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # 1. Stereo Depth Matcher (Computed ONCE per stereo frame)
        self.matcher = get_stereo_matcher(stereo_model)

        # 2. Pothole Segmentation Model
        if pothole_weights is None:
            pothole_weights = str(PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg.pt")
        self.pothole_model = YOLO(pothole_weights)
        self.conf_pothole = conf_pothole

        # 3. Pothole 3D Metrology Engine
        self.pothole_engine = PotholeGeometryEngine(
            min_depth_m=0.5,
            max_depth_m=45.0,
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

        # 5. Traffic Spatial Analyzer (Ray / Road-Plane Ground Contact)
        self.spatial_analyzer = TrafficSpatialAnalyzer(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            baseline=self.baseline, min_valid_disparity=0.5
        )

        # 6. Traffic Density Engine
        self.density_engine: Optional[TrafficDensityEngine] = None

        # 7. Crack 3D Geometry Engine
        lat_bound = float(calib.get("lateral_road_bound_m", 3.5 if float(calib.get("baseline", 0.12)) > 0.3 else 5.5))
        self.crack_engine = CrackGeometryEngine(lateral_road_bound_m=lat_bound)

        # 8. Evidence-Based Road Quality Engine
        self.road_quality_engine = RoadQualityEngine()

        # Crack segmentation model
        if crack_weights is None:
            cand = PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"
            if cand.exists():
                crack_weights = str(cand)
        self.crack_model = YOLO(crack_weights) if (crack_weights and Path(crack_weights).exists()) else None
        self.conf_crack = conf_crack

        # Temporal pothole association state
        self.next_pothole_id = 1
        self.active_potholes: Dict[int, Dict[str, Any]] = {}

    def _get_frame_calibration(self, H: int, W: int) -> Tuple[float, float, float, float, float]:
        """
        Validates that intrinsics correspond to the actual image coordinate system (W, H)
        and any resize/crop transforms.
        Supports arbitrary valid principal points within image bounds.
        Fails fast if calibration/image dimensions are inconsistent.
        """
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        b_val = self.baseline

        # If calib dict specifies source dimensions that differ from current frame
        c_w = self.calib.get("image_width", self.calib.get("width"))
        c_h = self.calib.get("image_height", self.calib.get("height"))
        if c_w is not None and c_h is not None:
            c_w = float(c_w)
            c_h = float(c_h)
            if c_w > 0 and c_h > 0 and (c_w != W or c_h != H):
                sx = float(W) / c_w
                sy = float(H) / c_h
                fx *= sx
                fy *= sy
                cx *= sx
                cy *= sy

        # Validate bounds
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Focal length must be positive: fx={fx}, fy={fy}")
        if b_val <= 0:
            raise ValueError(f"Stereo baseline must be positive: baseline={b_val}")
        if not (0.0 <= cx <= W):
            raise ValueError(f"Principal point cx={cx:.1f} out of bounds for image width {W}")
        if not (0.0 <= cy <= H):
            raise ValueError(f"Principal point cy={cy:.1f} out of bounds for image height {H}")

        return fx, fy, cx, cy, b_val

    def reset(self):
        """Resets temporal states (e.g. at sequence transitions)."""
        self.vehicle_tracker.reset()
        if self.density_engine:
            self.density_engine.reset()
        self.next_pothole_id = 1
        self.active_potholes.clear()
        self.rng = np.random.RandomState(self.random_seed)

    def _compute_stereo_quality(self, disparity: np.ndarray, depth_m: np.ndarray) -> Dict[str, Any]:
        """Calculates stereo quality metrics from the single stereo pass."""
        total_pixels = disparity.size
        valid_disp = (disparity > 0.5) & np.isfinite(disparity)
        valid_depth = (depth_m >= 1.0) & (depth_m <= 50.0) & np.isfinite(depth_m)

        valid_disp_frac = float(np.count_nonzero(valid_disp) / max(1, total_pixels))
        valid_depth_frac = float(np.count_nonzero(valid_depth) / max(1, total_pixels))

        med_disp = float(np.median(disparity[valid_disp])) if np.any(valid_disp) else 0.0
        med_depth = float(np.median(depth_m[valid_depth])) if np.any(valid_depth) else 0.0
        min_disp = float(np.min(disparity[valid_disp])) if np.any(valid_disp) else 0.0
        max_disp = float(np.max(disparity[valid_disp])) if np.any(valid_disp) else 0.0

        status = "OK" if valid_disp_frac >= 0.20 else "DEGRADED"

        return {
            "valid_disp_fraction": round(valid_disp_frac, 4),
            "valid_depth_fraction": round(valid_depth_frac, 4),
            "median_disparity_px": round(med_disp, 2),
            "median_depth_m": round(med_depth, 2),
            "min_disparity_px": round(min_disp, 2),
            "max_disparity_px": round(max_disp, 2),
            "status": status
        }

    def _estimate_general_road_plane(self, depth_m: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Fits a local road plane using RANSAC from valid depth points in the lower 35% of the image.
        Uses deterministic subsampling for 100% reproducible regression testing.
        """
        H, W = depth_m.shape[:2]
        road_roi_y = int(H * 0.65)
        roi_depth = depth_m[road_roi_y:H, :]
        valid = (roi_depth >= 1.0) & (roi_depth <= 35.0) & np.isfinite(roi_depth)
        
        fallback_meta = {
            "equation": None,
            "source": "none",
            "inlier_ratio": 0.0,
            "is_valid": False,
            "road_height_m": None,
            "pitch_deg": None,
            "roll_deg": None
        }

        if np.sum(valid) < 50:
            return None, fallback_meta

        ys, xs = np.where(valid)
        n_pts = len(ys)
        if n_pts > 1000:
            # Deterministic selection for reproducibility
            step = n_pts / 1000.0
            indices = np.array([int(i * step) for i in range(1000)], dtype=np.int32)
            ys = ys[indices]
            xs = xs[indices]

        ys_global = ys + road_roi_y
        zs = roi_depth[ys, xs]
        xs_3d = (xs - self.cx) * zs / self.fx
        ys_3d = (ys_global - self.cy) * zs / self.fy

        pts_3d = np.stack([xs_3d, ys_3d, zs], axis=1)

        from sklearn.linear_model import RANSACRegressor
        X_reg = pts_3d[:, [0, 2]]
        Y_reg = pts_3d[:, 1]
        try:
            ransac = RANSACRegressor(min_samples=10, residual_threshold=0.08, max_trials=100, random_state=self.random_seed)
            ransac.fit(X_reg, Y_reg)
            inlier_ratio = float(np.mean(ransac.inlier_mask_))
            if inlier_ratio < 0.35:
                # Retry with 0.15m threshold for rough / broken / potholed pavement
                ransac = RANSACRegressor(min_samples=10, residual_threshold=0.15, max_trials=100, random_state=self.random_seed)
                ransac.fit(X_reg, Y_reg)
                inlier_ratio = float(np.mean(ransac.inlier_mask_))
                if inlier_ratio < 0.30:
                    return None, fallback_meta
            m1, m2 = ransac.estimator_.coef_
            c_int = ransac.estimator_.intercept_
            plane = np.array([m1, -1.0, m2, c_int], dtype=np.float32)
            norm = np.linalg.norm(plane[:3])
            plane /= norm

            # Compute physical angles and height
            a, b, c, d = plane
            road_height = float(abs(d / b)) if abs(b) > 1e-4 else 1.65
            pitch_deg = float(np.degrees(np.arctan2(c, -b)))
            roll_deg = float(np.degrees(np.arctan2(a, -b)))

            meta = {
                "equation": [round(float(v), 5) for v in plane],
                "source": "ransac_road_surface",
                "inlier_ratio": round(inlier_ratio, 3),
                "is_valid": True,
                "road_height_m": round(road_height, 2),
                "pitch_deg": round(pitch_deg, 2),
                "roll_deg": round(roll_deg, 2)
            }
            return plane, meta
        except Exception:
            return None, fallback_meta

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

            is_valid = (pot.quality in ["GOOD", "WARNING"])
            record = {
                "track_id": assigned_id,
                "confidence": round(float(pot.confidence), 3),
                "bbox": [int(v) for v in pot.bbox_xyxy],
                "mask": pot.mask,
                "contour": pot.contour_px,
                "depth_p95_cm": round(float(pot.depth_p95_cm), 1) if (is_valid and pot.depth_p95_cm is not None) else None,
                "depth_median_cm": round(float(pot.depth_median_cm), 1) if (is_valid and pot.depth_median_cm is not None) else None,
                "depth_max_cm": round(float(pot.depth_max_cm), 1) if (is_valid and pot.depth_max_cm is not None) else None,
                "area_m2": round(float(pot.area_m2), 4) if (is_valid and pot.area_m2 is not None) else None,
                "length_m": round(float(pot.length_m), 2) if (is_valid and pot.length_m is not None) else None,
                "width_m": round(float(pot.width_m), 2) if (is_valid and pot.width_m is not None) else None,
                "center_3d": [round(float(v), 2) for v in pot.center_3d] if pot.center_3d is not None else None,
                "plane_equation": [round(float(v), 5) for v in pot.plane_equation] if pot.plane_equation is not None else None,
                "quality": pot.quality,
                "quality_reason": pot.quality_reason,
                "raw_area_m2": round(float(pot.raw_area_m2), 4) if pot.raw_area_m2 is not None else None,
                "raw_depth_p95_cm": round(float(pot.raw_depth_p95_cm), 1) if pot.raw_depth_p95_cm is not None else None
            }
            self.active_potholes[assigned_id] = record
            tracked.append(record)

        self.active_potholes = {k: v for k, v in self.active_potholes.items() if k in assigned_active or len(tracked) == 0}
        return tracked

    def process_frame(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        frame_index: int = 0,
        timestamp: Optional[float] = None
    ) -> UnifiedFrameState:
        """
        Executes one full unified stereo perception cycle:
        1. Single Stereo Pass -> disparity & metric depth
        2. Stereo Quality Evaluation
        3. Pothole Segmentation & 3D Shoelace Footprint Metrology Branch
        4. Road Plane Estimation (from pothole ring or RANSAC road surface)
        5. Vehicular Detection + ByteTrack + 3D Ground-Contact Localization Branch
        6. Pairwise & Ego Spatial Distance Engine
        7. Explicit 4-Vertex 3D Planar Road Polygon Traffic Density Engine
        8. Packages everything into UnifiedFrameState

        Returns:
            UnifiedFrameState object
        """
        t0 = time.time()
        H, W = left_rgb.shape[:2]
        if timestamp is None:
            timestamp = float(frame_index) / 10.0  # Default 10 fps timebase

        fx, fy, cx, cy, b_val = self._get_frame_calibration(H, W)

        if self.density_engine is None:
            self.density_engine = TrafficDensityEngine(
                fx=fx, img_width=W, img_height=H, z_min=2.0, z_max=35.0
            )

        # -------------------------------------------------------------
        # STEP 1: ONE STEREO PASS (Shared by all branches)
        # -------------------------------------------------------------
        t_stereo0 = time.time()
        disparity = self.matcher.estimate_disparity(left_rgb, right_rgb)
        valid_disp = (disparity > 0.5) & np.isfinite(disparity)
        depth_m = np.zeros_like(disparity, dtype=np.float32)
        depth_m[valid_disp] = (fx * b_val) / disparity[valid_disp]
        t_stereo_ms = (time.time() - t_stereo0) * 1000.0

        # Step 2: Stereo Quality
        stereo_quality = self._compute_stereo_quality(disparity, depth_m)

        # Estimate general road plane early for fallback support
        general_plane, general_plane_meta = self._estimate_general_road_plane(depth_m)

        # -------------------------------------------------------------
        # STEP 3: POTHOLE BRANCH (YOLO11-seg + 3D Metrology)
        # -------------------------------------------------------------
        t_pot_seg0 = time.time()
        pot_res = self.pothole_model(left_rgb, conf=self.conf_pothole, verbose=False)
        t_pot_seg_ms = (time.time() - t_pot_seg0) * 1000.0

        t_pot_geom0 = time.time()
        calib_dict = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": b_val}
        
        # Build individual instance dictionaries preserving confidence and exact contours
        pothole_yolo_list = []
        if pot_res[0].masks is not None and pot_res[0].boxes is not None:
            for idx, box in enumerate(pot_res[0].boxes):
                c_id = int(box.cls[0].item())
                if c_id == 0 or len(self.pothole_model.names) == 1:
                    xyxy = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    m = pot_res[0].masks.data[idx].cpu().numpy()
                    m_resized = cv2.resize((m > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    pothole_yolo_list.append({
                        "bbox": tuple(xyxy),
                        "confidence": float(box.conf[0]),
                        "mask": m_resized
                    })

        pot_instances_raw = self.pothole_engine.analyze_potholes(
            left_rgb, depth_m, pothole_yolo_list, calib_dict, fallback_road_plane=general_plane
        )
        tracked_potholes = self._track_potholes(pot_instances_raw)
        t_pot_geom_ms = (time.time() - t_pot_geom0) * 1000.0

        # -------------------------------------------------------------
        # STEP 4: ROAD PLANE ESTIMATION
        # -------------------------------------------------------------
        best_plane = None
        road_plane_meta = None
        for p in tracked_potholes:
            if p.get("plane_equation") is not None and p["quality"] in ["GOOD", "WARNING"]:
                best_plane = np.array(p["plane_equation"], dtype=np.float32)
                road_plane_meta = {
                    "equation": [round(float(v), 5) for v in best_plane],
                    "source": "pothole_ring",
                    "inlier_ratio": 1.0,
                    "is_valid": True,
                    "road_height_m": round(float(abs(best_plane[3] / best_plane[1])), 2) if abs(best_plane[1]) > 1e-4 else 1.65,
                    "pitch_deg": round(float(np.degrees(np.arctan2(best_plane[2], -best_plane[1]))), 2),
                    "roll_deg": round(float(np.degrees(np.arctan2(best_plane[0], -best_plane[1]))), 2)
                }
                break

        if best_plane is None:
            best_plane = general_plane
            road_plane_meta = general_plane_meta

        # -------------------------------------------------------------
        # STEP 5: TRAFFIC BRANCH (YOLO + ByteTrack + 3D Ground Contact)
        # -------------------------------------------------------------
        t_veh0 = time.time()
        tracked_vehicles = self.vehicle_tracker.track(left_rgb, frame_index=frame_index)
        t_veh_track_ms = (time.time() - t_veh0) * 1000.0

        t_spatial0 = time.time()
        traffic_scene = self.spatial_analyzer.analyze_scene(
            tracked_vehicles, disparity, road_plane=best_plane
        )
        t_spatial_ms = (time.time() - t_spatial0) * 1000.0

        t_density0 = time.time()
        density_telemetry = self.density_engine.update(
            scene_analysis=traffic_scene,
            road_plane=best_plane
        )
        t_density_ms = (time.time() - t_density0) * 1000.0

        total_time_ms = (time.time() - t0) * 1000.0
        fps = 1000.0 / max(1e-3, total_time_ms)

        # -------------------------------------------------------------
        # STEP 6: COMPOSE UNIFIED TRAFFIC METRICS & VEHICLE RECORDS
        # -------------------------------------------------------------
        ego_m = traffic_scene.get("ego_metrics", {})
        road_area = density_telemetry.get("observable_road_area_m2")
        if road_area is None:
            road_area = density_telemetry.get("road_polygon_area_m2")

        traffic_metrics = {
            "visible_vehicle_count": density_telemetry.get("visible_vehicle_count", 0),
            "tracked_vehicle_count": density_telemetry.get("tracked_vehicle_count", 0),
            "nearest_vehicle_distance_m": ego_m.get("nearest_vehicle_distance_m"),
            "nearest_vehicle_class": ego_m.get("nearest_vehicle_class"),
            "nearest_vehicle_id": ego_m.get("nearest_vehicle_id"),
            "nearest_longitudinal_gap_m": ego_m.get("nearest_longitudinal_gap_m"),
            "average_longitudinal_gap_m": ego_m.get("average_longitudinal_gap_m"),
            "traffic_density_veh_km2": density_telemetry.get("traffic_density_veh_km2"),
            "traffic_density_veh_m2": density_telemetry.get("traffic_density_veh_m2"),
            "traffic_level": density_telemetry.get("traffic_level", "NORMAL"),
            "traffic_level_raw": density_telemetry.get("traffic_level_raw", "NORMAL"),
            "road_polygon_area_m2": road_area,
            "observable_road_area_m2": road_area,
            "density_status": density_telemetry.get("density_status", "DENSITY: UNRELIABLE"),
            "pairwise_distances": traffic_scene.get("pairwise_distances", [])
        }

        # Build clean vehicle objects
        formatted_vehicles = []
        for v in traffic_scene.get("vehicles", []):
            formatted_vehicles.append({
                "track_id": v.get("track_id"),
                "display_id": v.get("display_id"),
                "class": v.get("class"),
                "confidence": round(float(v.get("confidence", 1.0)), 3),
                "bbox": [int(coord) for coord in v.get("bbox", [0, 0, 0, 0])],
                "ground_contact_px": v.get("ground_contact_px"),
                "X_m": v.get("X_m"),
                "Y_m": v.get("Y_m"),
                "Z_m": v.get("Z_m"),
                "distance_m": v.get("distance_m"),
                "longitudinal_gap_m": v.get("longitudinal_gap_m"),
                "position_method": v.get("position_method", "stereo"),
                "status": v.get("status", "OK")
            })

        # -------------------------------------------------------------
        # STEP 7: CRACK BRANCH (Segmentation + 3D Geometry)
        # -------------------------------------------------------------
        formatted_cracks = []
        t_crack_ms = 0.0
        obs_area_m2 = road_area

        if self.crack_model is not None:
            t_cr0 = time.time()
            cr_res = self.crack_model(left_rgb, conf=self.conf_crack, verbose=False)
            crack_mask = np.zeros((H, W), dtype=np.uint8)

            ref_crack_plane = best_plane if best_plane is not None else np.array([0.0, -1.0, 0.0, 1.65], dtype=np.float32)
            pa, pb, pc, pd = ref_crack_plane
            v_horiz = float(cy - fy * (pc / pb)) if abs(pb) > 1e-4 else cy

            if cr_res[0].masks is not None and cr_res[0].boxes is not None:
                for idx, box in enumerate(cr_res[0].boxes):
                    c_id = int(box.cls[0].item())
                    # In 2-class model (0: pothole, 1: crack)
                    if c_id == 1 or len(self.crack_model.names) == 1:
                        b_coords = box.xyxy[0].tolist()
                        box_area = (b_coords[2] - b_coords[0]) * (b_coords[3] - b_coords[1])
                        # Reject macro-scene false positives (>35% of screen)
                        if box_area > 0.35 * (W * H):
                            continue
                        # Top banner / UI exclusion: cannot start inside HUD banner
                        if b_coords[1] < 70:
                            continue
                        m_raw = cr_res[0].masks.data[idx].cpu().numpy()
                        m_resized = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                        # Reject macro-mask artifacts (>5% of screen)
                        if cv2.countNonZero(m_resized) > 0.05 * (W * H):
                            continue
                        crack_mask = np.bitwise_or(crack_mask, m_resized * 255)

            crack_instances = self.crack_engine.analyze_cracks(
                crack_mask=crack_mask,
                road_plane=ref_crack_plane,
                calib=calib_dict,
                observable_road_area_m2=obs_area_m2
            )
            for ci in crack_instances:
                formatted_cracks.append({
                    "track_id": ci.track_id,
                    "confidence": ci.confidence,
                    "bbox": ci.bbox_xyxy,
                    "mask": ci.mask,
                    "length_m": ci.length_m,
                    "coverage_area_m2": ci.coverage_area_m2,
                    "center_3d": ci.center_3d,
                    "linear_density_m_m2": ci.linear_density_m_m2,
                    "status": ci.status
                })
            t_crack_ms = (time.time() - t_cr0) * 1000.0

        # -------------------------------------------------------------
        # STEP 8: EVIDENCE-BASED ROAD QUALITY ENGINE
        # -------------------------------------------------------------
        t_rq0 = time.time()
        rq_res = self.road_quality_engine.evaluate_road_quality(
            potholes=tracked_potholes,
            cracks=formatted_cracks,
            road_plane_meta=road_plane_meta,
            depth_m=depth_m,
            calib=calib_dict,
            observable_road_area_m2=obs_area_m2,
            stereo_quality=stereo_quality
        )
        road_quality_dict = rq_res.to_dict()
        t_rq_ms = (time.time() - t_rq0) * 1000.0

        total_time_ms = (time.time() - t0) * 1000.0
        fps = 1000.0 / max(1e-3, total_time_ms)

        performance_metrics = {
            "stereo_ms": round(t_stereo_ms, 1),
            "pothole_seg_ms": round(t_pot_seg_ms, 1),
            "pothole_geom_ms": round(t_pot_geom_ms, 1),
            "pothole_total_ms": round(t_pot_seg_ms + t_pot_geom_ms, 1),
            "crack_ms": round(t_crack_ms, 1),
            "road_quality_ms": round(t_rq_ms, 1),
            "vehicle_tracking_ms": round(t_veh_track_ms, 1),
            "spatial_ms": round(t_spatial_ms, 1),
            "density_ms": round(t_density_ms, 1),
            "total_latency_ms": round(total_time_ms, 1),
            "effective_fps": round(fps, 1)
        }

        return UnifiedFrameState(
            frame_id=frame_index,
            timestamp=timestamp,
            vehicles=formatted_vehicles,
            potholes=tracked_potholes,
            cracks=formatted_cracks,
            road_quality=road_quality_dict,
            traffic_metrics=traffic_metrics,
            road_plane=road_plane_meta,
            stereo_quality=stereo_quality,
            performance_metrics=performance_metrics,
            disparity=disparity,
            depth_m=depth_m
        )


# Backward compatibility alias
BusSensePerceptionPipeline = UnifiedBusSensePipeline
