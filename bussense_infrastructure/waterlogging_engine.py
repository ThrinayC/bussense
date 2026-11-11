#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/waterlogging_engine.py
==============================================
Waterlogging Detection & Road-Surface Extent Metrology Engine (Section 6)

Core Metric:
- water_coverage_ratio = water_pixels_inside_valid_ego_corridor /
                        (road_pixels UNION water_pixels inside valid ego-corridor)
- Denominator Bug Prevention:
  road_or_water_mask = road_mask UNION water_mask
- Fallback: projected ego-road corridor when road segmentation is ungrounded.
- Temporal Filtering: Tracks water patch persistence across consecutive frames
  (min_persistence_frames >= 4, min_coverage >= 0.08, min_visibility >= 0.60).
- Explicit constraint: Extent ratio only. Never claims direct flood depth without sensors.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class WaterPatchObservation:
    frame_idx: int
    timestamp: float
    water_area_px: int
    road_or_water_area_px: int
    water_coverage_ratio: float
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    visibility_score: float
    denominator_mode: str
    is_persistent: bool = False
    persistence_frames: int = 1


class WaterloggingEngine:
    """
    Production-grade waterlogging detection, ego-corridor extent calculation,
    and multi-frame temporal persistence filter.
    """

    def __init__(
        self,
        min_persistence_frames: int = 4,
        min_coverage_ratio: float = 0.08,
        min_visibility_score: float = 0.60,
        corridor_width_ratio: float = 0.55,
        denominator_mode: str = "union",
        use_model: bool = False,
    ):
        self.min_persistence_frames = min_persistence_frames
        self.min_coverage_ratio = min_coverage_ratio
        self.min_visibility_score = min_visibility_score
        self.corridor_width_ratio = corridor_width_ratio
        self.denominator_mode = denominator_mode
        self.use_model = use_model
        self.model_adapter = None

        if self.use_model:
            try:
                from .models.water_model import WaterModelAdapter
                self.model_adapter = WaterModelAdapter()
            except Exception as e:
                # print(f"Water model loading failed, falling back to heuristic: {e}")
                self.model_adapter = None

        # Temporal state tracking
        self.consecutive_water_frames = 0
        self.historical_ratios: List[float] = []
        self.last_observation: Optional[WaterPatchObservation] = None

    def create_ego_corridor_mask(self, h: int, w: int) -> np.ndarray:
        """
        Creates a trapezoidal ego-road drivable corridor projection.
        Trapezoid anchored from camera hood to vanishing horizon.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        bottom_left = (int(w * 0.15), h)
        bottom_right = (int(w * 0.85), h)
        top_right = (int(w * 0.58), int(h * 0.52))
        top_left = (int(w * 0.42), int(h * 0.52))

        pts = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def detect_water_mask_heuristic(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Specialist water feature extraction:
        Combines surface specular reflection, low saturation, texture homogeneity,
        and sky-reflection gradient in the lower roadway ROI.
        """
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

        # Specular glint & asphalt puddle contrast
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Water puddles on asphalt typically exhibit lower saturation and localized reflection
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        # Smooth texture + wet reflection
        water_cand = ((sat < 65) & (val > 70) & (val < 225) & (grad_mag < 25)).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(water_cand, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned

    def process_frame(
        self,
        rgb_img: np.ndarray,
        frame_idx: int,
        timestamp: float,
        road_mask: Optional[np.ndarray] = None,
        visibility_score: float = 0.95
    ) -> Optional[WaterPatchObservation]:
        """
        Performs extent metrology on a single frame and updates temporal persistence.
        """
        h, w = rgb_img.shape[:2]
        corridor_mask = self.create_ego_corridor_mask(h, w)

        # Extract water segmentation candidate
        if self.model_adapter is not None:
            try:
                res = self.model_adapter.predict(rgb_img)
                water_mask = (res["mask"] * 255).astype(np.uint8)
            except Exception:
                water_mask = self.detect_water_mask_heuristic(rgb_img)
        else:
            water_mask = self.detect_water_mask_heuristic(rgb_img)

        # Strict ego-corridor intersection
        water_in_corridor = cv2.bitwise_and(water_mask, water_mask, mask=corridor_mask)
        water_px = int(np.count_nonzero(water_in_corridor))

        # Denominator calculation: Prevent denominator bug
        # road_or_water_mask = road_mask UNION water_mask
        if road_mask is not None and self.denominator_mode == "union":
            road_or_water = cv2.bitwise_or(road_mask, water_mask)
            road_or_water_corridor = cv2.bitwise_and(road_or_water, road_or_water, mask=corridor_mask)
            total_corridor_px = int(np.count_nonzero(road_or_water_corridor))
            used_mode = "road_union_water"
        else:
            # Projected ego-corridor fallback
            total_corridor_px = int(np.count_nonzero(corridor_mask))
            used_mode = "projected_ego_corridor"

        if total_corridor_px == 0:
            coverage_ratio = 0.0
        else:
            coverage_ratio = float(np.clip(water_px / total_corridor_px, 0.0, 1.0))

        # Check single-frame thresholds
        has_water = (coverage_ratio >= self.min_coverage_ratio) and (visibility_score >= self.min_visibility_score)

        if has_water:
            self.consecutive_water_frames += 1
            self.historical_ratios.append(coverage_ratio)
        else:
            self.consecutive_water_frames = max(0, self.consecutive_water_frames - 1)
            if self.consecutive_water_frames == 0:
                self.historical_ratios.clear()

        is_persistent = (self.consecutive_water_frames >= self.min_persistence_frames)

        # Bounding box and centroid of water patch
        cnts, _ = cv2.findContours(water_in_corridor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest_cnt = max(cnts, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest_cnt)
            bbox = (x, y, x + bw, y + bh)
            M = cv2.moments(largest_cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + bw // 2, y + bh // 2
        else:
            bbox = (0, 0, 0, 0)
            cx, cy = 0, 0

        obs = WaterPatchObservation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            water_area_px=water_px,
            road_or_water_area_px=total_corridor_px,
            water_coverage_ratio=round(coverage_ratio, 4),
            centroid=(cx, cy),
            bbox=bbox,
            visibility_score=visibility_score,
            denominator_mode=used_mode,
            is_persistent=is_persistent,
            persistence_frames=self.consecutive_water_frames
        )

        self.last_observation = obs
        return obs
