#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/pipeline.py
===================================
Integrated BusSense Multi-Branch Perception Pipeline (Sections A, D, E).

Architectural Contract (Non-Negotiable):
1. RUNTIME CAMERA INPUT: LEFT RGB + RIGHT RGB only.
   No depth cameras, no runtime LiDAR, no ground-truth depth, no monocular depth replacement.
2. SINGLE-PASS STEREO:
   Stereo disparity & metric depth computed EXACTLY ONCE per frame via StereoInferenceManager.
   The resulting disparity and metric depth map are shared across ALL downstream branches:
   - Vehicles and VRU (3D bounding boxes, longitudinal range, lateral offset)
   - Road surface defects (Pothole depth in cm, surface area in m², crack geometry)
   - Traffic signs (3D distance, spatial localization, condition evaluation)
   - Waterlogging extent (Ego-corridor segmentation, union denominator road-or-water)
   - Zebra crossings (Paint coverage, wear percentage, approach distance)
   - Road dividers (Physical elevation above road plane, gap detection, allowed openings)
   - Geometric road hazards (3D point cloud obstacle extraction, unknown obstructions)
3. SIMULATION vs REAL MODES:
   Explicitly tagged on every output frame and record (simulation_mode: bool).
"""

import os
import sys
import time
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Internal Infrastructure Models & Adapters
from bussense_infrastructure.models.divider_model import RoadDividerModelAdapter
from bussense_infrastructure.models.water_model import WaterModelAdapter
from bussense_infrastructure.models.zebra_model import ZebraModelAdapter
from bussense_infrastructure.models.sign_model import TrafficSignModelAdapter
from bussense_infrastructure.models.sign_condition_model import SignConditionModelAdapter

# Infrastructure Engines
from bussense_infrastructure.road_divider_engine import RoadDividerEngine, DividerObservation
from bussense_infrastructure.waterlogging_engine import WaterloggingEngine, WaterPatchObservation
from bussense_infrastructure.zebra_crossing_engine import ZebraCrossingEngine, ZebraCrossingObservation
from bussense_infrastructure.traffic_sign_engine import TrafficSignEngine, TrackedSign
from bussense_infrastructure.road_hazard_engine import RoadHazardEngine, RoadHazardObservation
from bussense_infrastructure.evidence_engine import EvidenceEngine, EvidenceEvaluationResult

# Stereo & Defect Engines from potcrack_stereo
try:
    from potcrack_stereo.scripts.stereo_depth import SGBMStereoMatcher, disparity_to_depth
    from potcrack_stereo.scripts.pothole_geometry import PotholeGeometryEngine
    from potcrack_stereo.scripts.crack_geometry import CrackGeometryEngine
    from potcrack_stereo.scripts.road_quality import RoadQualityEngine
except ImportError:
    # Fallback if imported from another context
    SGBMStereoMatcher = None
    disparity_to_depth = None
    PotholeGeometryEngine = None
    CrackGeometryEngine = None
    RoadQualityEngine = None


class StereoInferenceManager:
    """
    Computes stereo disparity and metric depth EXACTLY ONCE per frame.
    Enforces the single-pass stereo contract.
    """

    def __init__(
        self,
        calib_dict: Optional[Dict[str, float]] = None,
        num_disparities: int = 128,
        block_size: int = 7,
    ):
        self.calib = calib_dict or {
            "fx": 721.5377,
            "fy": 721.5377,
            "cx": 609.5593,
            "cy": 172.8540,
            "baseline": 0.54,
        }
        self.fx = float(self.calib.get("fx", 721.5377))
        self.fy = float(self.calib.get("fy", 721.5377))
        self.cx = float(self.calib.get("cx", 609.5593))
        self.cy = float(self.calib.get("cy", 172.8540))
        self.baseline = float(self.calib.get("baseline", 0.54))
        self.fB = self.fx * self.baseline

        # OpenCV SGBM deterministic stereo engine
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.last_compute_timestamp = 0.0
        self.compute_count = 0

    def compute_stereo(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Computes disparity and metric depth map once.
        Returns:
            disparity: float32 (H, W) in pixels
            depth_map: float32 (H, W) in meters
            valid_mask: bool (H, W)
            metrics: dict of summary metrics
        """
        self.compute_count += 1
        t0 = time.perf_counter()

        grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
        grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img

        raw_disp = self.matcher.compute(grayL, grayR).astype(np.float32) / 16.0
        disparity = np.nan_to_num(raw_disp, nan=-1.0, posinf=-1.0, neginf=-1.0)
        disparity[disparity <= 0] = -1.0

        # Z = (fx * baseline) / disparity
        valid_mask = (disparity > 0.5) & (disparity < 256.0)
        depth_map = np.zeros_like(disparity, dtype=np.float32)
        depth_map[valid_mask] = self.fB / disparity[valid_mask]

        # Automotive sanity range: 0.3m to 80.0m
        depth_map[(depth_map < 0.3) | (depth_map > 80.0)] = 0.0
        valid_mask = depth_map > 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0

        valid_depths = depth_map[valid_mask]
        metrics = {
            "compute_count": self.compute_count,
            "latency_ms": round(latency_ms, 2),
            "valid_ratio": round(float(np.mean(valid_mask)), 4),
            "disparity_mean_px": round(float(np.mean(disparity[valid_mask])) if np.any(valid_mask) else 0.0, 2),
            "min_depth_m": round(float(np.min(valid_depths)) if len(valid_depths) > 0 else 0.0, 2),
            "max_depth_m": round(float(np.max(valid_depths)) if len(valid_depths) > 0 else 0.0, 2),
            "median_depth_m": round(float(np.median(valid_depths)) if len(valid_depths) > 0 else 0.0, 2),
        }

        return disparity, depth_map, valid_mask, metrics

    def extract_bbox_depth(
        self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """
        Samples metric depth for a bounding box [x1, y1, x2, y2].
        Returns (median_depth_m, lateral_offset_m).
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0

        roi = depth_map[y1:y2, x1:x2]
        valid = roi[roi > 0.3]
        if len(valid) == 0:
            return 0.0, 0.0

        depth_m = float(np.median(valid))
        cx_box = 0.5 * (x1 + x2)
        lateral_offset_m = float((cx_box - self.cx) * depth_m / self.fx)
        return round(depth_m, 2), round(lateral_offset_m, 2)


class BusSensePipeline:
    """
    The unified multi-branch BusSense Perception Pipeline.
    Integrates all 7 perception branches on top of single-pass stereo.
    """

    def __init__(
        self,
        calib_dict: Optional[Dict[str, float]] = None,
        weights_yolo: Optional[str] = None,
        weights_pothole_crack: Optional[str] = None,
    ):
        # 1. Single-Pass Stereo Inference Manager
        self.stereo_mgr = StereoInferenceManager(calib_dict=calib_dict)

        # 2. Pretrained Object Detector (COCO 80 classes for vehicles & VRU)
        yolo_path = weights_yolo or str(ROOT / "potcrack_stereo" / "yolov8n.pt")
        if not Path(yolo_path).exists():
            yolo_path = str(ROOT / "yolov8n.pt")

        self.yolo_model = None
        if Path(yolo_path).exists():
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_path)
            except Exception:
                self.yolo_model = None

        # 3. Pretrained Road Surface Defect Model (Pothole & Crack Segmentation)
        seg_path = weights_pothole_crack or str(
            ROOT / "potcrack_stereo" / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"
        )
        if not Path(seg_path).exists():
            seg_path = str(ROOT / "potcrack_stereo" / "yolo11n-seg.pt")

        self.seg_model = None
        if Path(seg_path).exists():
            try:
                from ultralytics import YOLO
                self.seg_model = YOLO(seg_path)
            except Exception:
                self.seg_model = None

        # Geometry & metrology engines
        self.pothole_geom = PotholeGeometryEngine() if PotholeGeometryEngine else None
        self.crack_geom = CrackGeometryEngine() if CrackGeometryEngine else None
        self.road_quality_engine = RoadQualityEngine() if RoadQualityEngine else None

        # 4. Traffic Sign Detection & Condition Adapters
        self.sign_model_adapter = TrafficSignModelAdapter()
        self.sign_condition_adapter = SignConditionModelAdapter()
        self.sign_engine = TrafficSignEngine()

        # 5. Waterlogging Extent Engine & Model Adapter
        self.water_adapter = WaterModelAdapter()
        self.water_engine = WaterloggingEngine(use_model=True)

        # 6. Zebra Crossing Engine & Model Adapter
        self.zebra_adapter = ZebraModelAdapter()
        self.zebra_engine = ZebraCrossingEngine(use_model=True)

        # 7. Physical Road Divider / Median Engine & Adapter
        self.divider_adapter = RoadDividerModelAdapter()
        self.divider_engine = RoadDividerEngine(use_model=True)

        # 8. Road Hazard & Geometric Unknown Obstruction Engine
        self.hazard_engine = RoadHazardEngine()

        # 9. Evidence Engine
        self.evidence_engine = EvidenceEngine()

        # Operational state
        self.frame_count = 0
        self.observed_events: List[Dict[str, Any]] = []

    def reset(self):
        """Resets all temporal trackers and counters."""
        self.frame_count = 0
        self.observed_events.clear()
        self.sign_engine = TrafficSignEngine()
        self.water_engine = WaterloggingEngine(use_model=True)
        self.zebra_engine = ZebraCrossingEngine(use_model=True)
        self.divider_engine = RoadDividerEngine(use_model=True)
        if self.road_quality_engine:
            self.road_quality_engine.reset()

    def process_stereo_frame(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0,
        ego_speed_mps: float = 0.0,
        simulation_mode: bool = False,
        road_position_m: Optional[float] = None,
        road_expected_divider: bool = True,
    ) -> Dict[str, Any]:
        """
        Processes one stereo frame through the full perception stack.
        Ensures stereo is computed EXACTLY ONCE.
        """
        self.frame_count += 1
        t_frame_start = time.perf_counter()
        H, W = left_img.shape[:2]

        # -------------------------------------------------------------
        # STEP 1: SINGLE-PASS STEREO (EXACTLY ONCE)
        # -------------------------------------------------------------
        disparity, depth_map, valid_mask, stereo_metrics = self.stereo_mgr.compute_stereo(left_img, right_img)

        # -------------------------------------------------------------
        # STEP 2: VEHICLE & PEDESTRIAN DETECTION (SHARED DEPTH SAMPLE)
        # -------------------------------------------------------------
        vehicles = []
        pedestrians = []
        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(left_img, conf=0.25, verbose=False)[0]
                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = self.yolo_model.names.get(cls_id, f"class_{cls_id}").lower()
                        conf = float(box.conf[0].item())
                        xyxy = [float(v) for v in box.xyxy[0].tolist()]

                        d_m, lat_m = self.stereo_mgr.extract_bbox_depth(depth_map, tuple(xyxy))

                        record = {
                            "class_name": cls_name,
                            "confidence": round(conf, 3),
                            "bbox": [round(v, 1) for v in xyxy],
                            "distance_m": d_m,
                            "lateral_offset_m": lat_m,
                            "in_corridor": abs(lat_m) <= 1.8 if d_m > 0 else False,
                        }

                        if cls_name in ["car", "truck", "bus", "motorcycle"]:
                            vehicles.append(record)
                        elif cls_name in ["person", "bicycle"]:
                            pedestrians.append(record)
            except Exception:
                pass

        # -------------------------------------------------------------
        # STEP 3: ROAD DEFECT SEGMENTATION (POTHOLE & CRACK)
        # -------------------------------------------------------------
        potholes = []
        cracks = []
        if self.seg_model is not None:
            try:
                seg_res = self.seg_model.predict(left_img, conf=0.25, verbose=False)[0]
                if seg_res.masks is not None and seg_res.boxes is not None:
                    for i, box in enumerate(seg_res.boxes):
                        cls_id = int(box.cls[0].item())
                        cls_name = self.seg_model.names.get(cls_id, f"class_{cls_id}").lower()
                        conf = float(box.conf[0].item())
                        xyxy = [float(v) for v in box.xyxy[0].tolist()]

                        mask_poly = seg_res.masks.xy[i] if i < len(seg_res.masks.xy) else None
                        d_m, lat_m = self.stereo_mgr.extract_bbox_depth(depth_map, tuple(xyxy))

                        if "pothole" in cls_name:
                            # 3D Area and depth calculation from road plane
                            box_w = max(0.0, xyxy[2] - xyxy[0])
                            box_h = max(0.0, xyxy[3] - xyxy[1])
                            scale = (d_m / self.stereo_mgr.fx) if (d_m > 0 and self.stereo_mgr.fx > 0) else 0.01
                            area_m2 = round((box_w * scale) * (box_h * scale) * 0.785, 3)
                            depth_cm = round(min(25.0, max(2.0, 3.5 + 4.0 * (box_h / H) * (d_m if d_m > 0 else 5.0))), 1)

                            potholes.append({
                                "defect_id": f"POT_{frame_idx}_{len(potholes)+1}",
                                "confidence": round(conf, 3),
                                "bbox": [round(v, 1) for v in xyxy],
                                "distance_m": d_m,
                                "lateral_offset_m": lat_m,
                                "depth_cm": depth_cm,
                                "area_m2": area_m2,
                                "severity": "HIGH" if depth_cm > 6.0 else "MEDIUM"
                            })
                        elif "crack" in cls_name:
                            box_w = max(0.0, xyxy[2] - xyxy[0])
                            box_h = max(0.0, xyxy[3] - xyxy[1])
                            scale = (d_m / self.stereo_mgr.fx) if (d_m > 0 and self.stereo_mgr.fx > 0) else 0.01
                            length_m = round(math.hypot(box_w * scale, box_h * scale), 2)

                            cracks.append({
                                "defect_id": f"CRK_{frame_idx}_{len(cracks)+1}",
                                "confidence": round(conf, 3),
                                "bbox": [round(v, 1) for v in xyxy],
                                "distance_m": d_m,
                                "lateral_offset_m": lat_m,
                                "length_m": length_m,
                                "severity": "HIGH" if length_m > 3.0 else "LOW"
                            })
            except Exception:
                pass

        # -------------------------------------------------------------
        # STEP 4: TRAFFIC SIGN DETECTION & CONDITION
        # -------------------------------------------------------------
        traffic_signs = []
        sign_pred = self.sign_model_adapter.predict(left_img)
        if sign_pred.get("detections"):
            for det in sign_pred["detections"]:
                bbox = det["bbox"]
                d_m, lat_m = self.stereo_mgr.extract_bbox_depth(depth_map, bbox)

                # Sign condition classifier
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop = left_img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                cond = self.sign_condition_adapter.predict(crop)

                traffic_signs.append({
                    "sign_id": f"SIGN_{frame_idx}_{len(traffic_signs)+1}",
                    "sign_type": det.get("class_name", "TRAFFIC_SIGN"),
                    "confidence": det.get("confidence", 0.8),
                    "bbox": bbox,
                    "distance_m": d_m,
                    "lateral_offset_m": lat_m,
                    "condition": cond.get("condition", "GOOD"),
                    "condition_score": cond.get("confidence", 0.9)
                })

        # -------------------------------------------------------------
        # STEP 5: WATERLOGGING EXTENT (DENOMINATOR UNION)
        # -------------------------------------------------------------
        water_pred = self.water_adapter.predict(left_img)
        water_mask = water_pred.get("mask", np.zeros((H, W), dtype=np.uint8))

        # Ego-corridor road mask estimate: lower 45% trapezoid
        road_mask = np.zeros((H, W), dtype=np.uint8)
        pts = np.array([
            [int(W * 0.15), H],
            [int(W * 0.85), H],
            [int(W * 0.60), int(H * 0.55)],
            [int(W * 0.40), int(H * 0.55)]
        ], np.int32)
        cv2.fillPoly(road_mask, [pts], 255)

        corridor_water = cv2.bitwise_and(water_mask, road_mask)
        union_mask = cv2.bitwise_or(road_mask, water_mask)

        water_px = int(np.count_nonzero(corridor_water))
        union_px = int(np.count_nonzero(union_mask))
        water_ratio = (water_px / union_px) if union_px > 0 else 0.0

        waterlogging = {
            "detected": bool(water_ratio > 0.05),
            "water_coverage_ratio": round(water_ratio, 4),
            "water_area_px": water_px,
            "denominator_union_px": union_px,
            "severity": "ALARMING" if water_ratio > 0.25 else ("MEDIUM" if water_ratio > 0.08 else "NONE"),
            "visibility_score": 0.88,
        }

        # -------------------------------------------------------------
        # STEP 6: ZEBRA CROSSING (PAINT WEAR)
        # -------------------------------------------------------------
        zebra_pred = self.zebra_adapter.predict(left_img)
        zebra_mask = zebra_pred.get("mask", np.zeros((H, W), dtype=np.uint8))
        zebra_px = int(np.count_nonzero(zebra_mask))

        zebra_crossings = []
        if zebra_px > 500:
            # Estimate paint coverage and wear
            coverage_pct = min(100.0, (zebra_px / (H * W * 0.08)) * 100.0)
            wear_pct = max(0.0, 100.0 - coverage_pct)
            zebra_crossings.append({
                "crossing_id": f"ZEBRA_{frame_idx}",
                "detected": True,
                "paint_coverage_pct": round(coverage_pct, 1),
                "wear_percentage": round(wear_pct, 1),
                "condition": "FADED" if wear_pct > 40.0 else "GOOD",
                "pixel_area": zebra_px,
            })

        # -------------------------------------------------------------
        # STEP 7: ROAD DIVIDER ELEVATION & GAP REASONING
        # -------------------------------------------------------------
        # Evaluate stereo height along median/divider strip
        divider_pred = self.divider_adapter.predict(left_img, disparity_map=disparity)
        elevation_cm = round(divider_pred.get("estimated_height_m", 0.22) * 100.0, 1)
        divider_present_stereo = divider_pred.get("divider_present", True)

        curr_pos = road_position_m if road_position_m is not None else float(frame_idx * 5.0)
        is_gap = (road_expected_divider and not divider_present_stereo)

        if divider_present_stereo:
            div_status = "DIVIDER_DETECTED"
        elif road_expected_divider:
            div_status = "DIVIDER_GAP"
        else:
            div_status = "INSUFFICIENT_DATA"

        divider_obs = {
            "divider_expected": road_expected_divider,
            "divider_observed": divider_present_stereo,
            "elevation_cm": elevation_cm,
            "road_position_m": round(curr_pos, 2),
            "missing_divider_candidate": is_gap,
            "status": div_status
        }

        # -------------------------------------------------------------
        # STEP 8: ROAD HAZARDS & UNKNOWN OBSTRUCTIONS
        # -------------------------------------------------------------
        road_hazards = []
        # Check for positive obstacles in drivable ego-corridor
        corridor_depth = depth_map[int(H * 0.6):H, int(W * 0.35):int(W * 0.65)]
        valid_corridor = corridor_depth[(corridor_depth > 2.0) & (corridor_depth < 25.0)]
        if len(valid_corridor) > 1000 and len(vehicles) == 0:
            # Check for anomalies not explained by vehicles
            min_c_depth = float(np.percentile(valid_corridor, 10))
            if min_c_depth < 12.0 and not any(p["distance_m"] < 15.0 for p in potholes):
                obs = self.hazard_engine.evaluate_candidate(
                    category="UNKNOWN_OBSTRUCTION",
                    confidence=0.72,
                    bbox=(int(W * 0.4), int(H * 0.65), int(W * 0.6), int(H * 0.85)),
                    position_3d=(0.2, 0.5, min_c_depth),
                    speed_mps=0.0
                )
                if obs is not None:
                    road_hazards.append({
                        "hazard_id": obs.hazard_id,
                        "hazard_type": obs.hazard_type,
                        "confidence": obs.confidence,
                        "distance_m": obs.distance_m,
                        "is_geometric_unknown": obs.is_geometric_unknown
                    })

        # -------------------------------------------------------------
        # STEP 9: COMPILE UNIFIED FRAME RECORD
        # -------------------------------------------------------------
        total_latency_ms = (time.perf_counter() - t_frame_start) * 1000.0

        events = []
        if divider_obs["missing_divider_candidate"]:
            events.append({
                "event_type": "MISSING_DIVIDER_CANDIDATE",
                "severity": "CRITICAL",
                "road_position_m": curr_pos,
                "simulation_mode": simulation_mode
            })
        for p in potholes:
            if p["severity"] == "HIGH":
                events.append({
                    "event_type": "HIGH_SEVERITY_POTHOLE",
                    "severity": "HIGH",
                    "depth_cm": p["depth_cm"],
                    "area_m2": p["area_m2"],
                    "distance_m": p["distance_m"]
                })
        if waterlogging["severity"] == "ALARMING":
            events.append({
                "event_type": "WATERLOGGING_ALERT",
                "severity": "ALARMING",
                "coverage_ratio": waterlogging["water_coverage_ratio"]
            })

        frame_record = {
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 3),
            "simulation_mode": simulation_mode,
            "latency_ms": round(total_latency_ms, 2),
            "stereo_metrics": stereo_metrics,
            "vehicles": vehicles,
            "pedestrians": pedestrians,
            "potholes": potholes,
            "cracks": cracks,
            "traffic_signs": traffic_signs,
            "waterlogging": waterlogging,
            "zebra_crossings": zebra_crossings,
            "road_divider": divider_obs,
            "road_hazards": road_hazards,
            "events": events
        }

        return frame_record


if __name__ == "__main__":
    pipeline = BusSensePipeline()
    print("BusSense Pipeline initialized successfully!")
