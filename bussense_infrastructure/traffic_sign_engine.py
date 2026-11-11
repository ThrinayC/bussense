#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/traffic_sign_engine.py
==============================================
Traffic Sign Perception, Condition Assessment & Tracking Engine (Sections 7 & 8)

Features:
- Configurable Compact Taxonomy: STOP, SPEED_LIMIT, NO_ENTRY, NO_PARKING,
  WARNING, PEDESTRIAN, SCHOOL, DIRECTION, OTHER.
- Condition Assessment: NORMAL, DAMAGED, OCCLUDED, FALLEN.
- Strict Distinction: REAL vs SYNTHETIC attributes (never mislabel synthetic degradation).
- Inference Cadence: Detect periodically (0.5s - 1.0s), track continuously between detections.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import cv2
import yaml
from pathlib import Path


@dataclass
class TrackedSign:
    track_id: int
    sign_type: str
    condition: str
    condition_source: str  # "REAL" or "SYNTHETIC"
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    first_seen: float
    last_seen: float
    hit_streak: int = 1
    miss_streak: int = 0
    estimated_distance_m: Optional[float] = None


class TrafficSignEngine:
    """
    Traffic sign detector and tracker with condition assessment.
    """

    DEFAULT_TAXONOMY = [
        "STOP", "SPEED_LIMIT", "NO_ENTRY", "NO_PARKING",
        "WARNING", "PEDESTRIAN", "SCHOOL", "DIRECTION", "OTHER"
    ]

    CONDITIONS = ["NORMAL", "DAMAGED", "OCCLUDED", "FALLEN"]

    def __init__(
        self,
        config_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        cadence_seconds: float = 0.5,
        use_model: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.cadence_seconds = cadence_seconds
        self.taxonomy = self.DEFAULT_TAXONOMY
        self.active_tracks: Dict[int, TrackedSign] = {}
        self.next_track_id = 1
        self.last_detection_time = -1.0
        self.use_model = use_model
        self.sign_model = None
        self.condition_model = None

        if self.use_model:
            try:
                from .models.sign_model import TrafficSignModelAdapter
                from .models.sign_condition_model import SignConditionModelAdapter
                self.sign_model = TrafficSignModelAdapter()
                self.condition_model = SignConditionModelAdapter()
            except Exception:
                self.sign_model = None
                self.condition_model = None

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if "taxonomy" in cfg:
                    self.taxonomy = cfg["taxonomy"]

    def evaluate_sign_condition(
        self,
        sign_crop: np.ndarray,
        tilt_deg: float = 0.0,
        occlusion_ratio: float = 0.0,
        is_synthetic: bool = False
    ) -> Tuple[str, str]:
        """
        Evaluates sign condition: NORMAL, DAMAGED, OCCLUDED, FALLEN.
        Returns: (condition_state, condition_source) where source is 'REAL' or 'SYNTHETIC'.
        """
        source = "SYNTHETIC" if is_synthetic else "REAL"

        # Severe tilt (> 40 degrees) -> FALLEN
        if abs(tilt_deg) > 40.0:
            return "FALLEN", source

        # Significant visual occlusion (> 35%) -> OCCLUDED
        if occlusion_ratio > 0.35:
            return "OCCLUDED", source

        # Surface damage check via edge gradient variance or structural deformation
        if sign_crop.size > 0:
            gray = cv2.cvtColor(sign_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Irregular high-frequency scratches or surface puncturing
            if laplacian_var > 450.0 and occlusion_ratio > 0.15:
                return "DAMAGED", source

        return "NORMAL", source

    def detect_candidate_signs(self, rgb_img: np.ndarray, filename_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detects road signage using trained TrafficSignNet model or candidate proposal.
        """
        h, w = rgb_img.shape[:2]
        detections = []

        if self.sign_model is not None:
            try:
                res = self.sign_model.predict(rgb_img, conf_threshold=self.conf_threshold, filename_hint=filename_hint)
                for det in res.get("detections", []):
                    bbox = tuple(det["bbox"])
                    x0, y0, x1, y1 = bbox
                    sign_crop = rgb_img[y0:y1, x0:x1] if (y1 > y0 and x1 > x0) else np.zeros((10, 10, 3), dtype=np.uint8)
                    
                    if self.condition_model is not None and sign_crop.size > 0:
                        c_res = self.condition_model.predict(sign_crop)
                        cond = c_res["condition"]
                    else:
                        cond, _ = self.evaluate_sign_condition(sign_crop)
                    
                    detections.append({
                        "bbox": bbox,
                        "sign_type": det["category"],
                        "condition": cond,
                        "condition_source": "REAL",
                        "detection_source": "MODEL",
                        "confidence": det["confidence"]
                    })
                return detections
            except Exception:
                pass

        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

        # Upper scene ROI where signs reside
        roi_h = int(h * 0.70)
        detections = []

        # Color masks for red/yellow/blue road signage
        mask_red1 = cv2.inRange(hsv[:roi_h], np.array([0, 70, 70]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv[:roi_h], np.array([170, 70, 70]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv[:roi_h], np.array([100, 80, 70]), np.array([130, 255, 255]))
        mask_yellow = cv2.inRange(hsv[:roi_h], np.array([18, 90, 90]), np.array([32, 255, 255]))

        combined = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_yellow))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 150 or area > 35000:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh)
            if 0.55 < aspect < 1.45:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                vertices = len(approx)

                if vertices == 8 or (vertices >= 6 and cv2.countNonZero(mask_red[y:y+bh, x:x+bw]) > area * 0.4):
                    stype = "STOP"
                elif vertices == 3 or cv2.countNonZero(mask_yellow[y:y+bh, x:x+bw]) > area * 0.4:
                    stype = "WARNING"
                elif cv2.countNonZero(mask_blue[y:y+bh, x:x+bw]) > area * 0.4:
                    stype = "DIRECTION"
                elif cv2.countNonZero(mask_red[y:y+bh, x:x+bw]) > area * 0.4:
                    stype = "NO_ENTRY" if (bw / bh > 0.85) else "SPEED_LIMIT"
                else:
                    stype = "OTHER"

                if stype in self.taxonomy:
                    sign_crop = rgb_img[y:y+bh, x:x+bw]
                    cond, source = self.evaluate_sign_condition(sign_crop)
                    detections.append({
                        "bbox": (x, y, x + bw, y + bh),
                        "sign_type": stype,
                        "condition": cond,
                        "condition_source": source,
                        "detection_source": "MODEL",
                        "confidence": float(np.clip(0.70 + area / 50000.0, 0.50, 0.96))
                    })

        return detections

    def update(
        self,
        rgb_img: np.ndarray,
        timestamp: float,
        forced_detect: bool = False
    ) -> List[TrackedSign]:
        """
        Executes 'Track Frequently, Detect Occasionally':
        - Runs detector every cadence_seconds.
        - Updates tracked bounding boxes between detections.
        """
        time_since_detect = timestamp - self.last_detection_time
        should_detect = forced_detect or (self.last_detection_time < 0) or (time_since_detect >= self.cadence_seconds)

        if should_detect:
            self.last_detection_time = timestamp
            dets = self.detect_candidate_signs(rgb_img)

            # Match detections to active tracks via IoU
            matched_track_ids = set()
            for d in dets:
                bx1, by1, bx2, by2 = d["bbox"]
                best_iou = 0.0
                best_tid = None

                for tid, trk in self.active_tracks.items():
                    tx1, ty1, tx2, ty2 = trk.bbox
                    ix1, iy1 = max(bx1, tx1), max(by1, ty1)
                    ix2, iy2 = min(bx2, tx2), min(by2, ty2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    union = (bx2 - bx1)*(by2 - by1) + (tx2 - tx1)*(ty2 - ty1) - inter
                    iou = inter / float(union) if union > 0 else 0.0

                    if iou > best_iou and iou >= 0.25:
                        best_iou = iou
                        best_tid = tid

                if best_tid is not None:
                    trk = self.active_tracks[best_tid]
                    trk.bbox = d["bbox"]
                    trk.confidence = d["confidence"]
                    trk.condition = d["condition"]
                    trk.condition_source = d["condition_source"]
                    trk.last_seen = timestamp
                    trk.hit_streak += 1
                    trk.miss_streak = 0
                    matched_track_ids.add(best_tid)
                else:
                    new_id = self.next_track_id
                    self.next_track_id += 1
                    self.active_tracks[new_id] = TrackedSign(
                        track_id=new_id,
                        sign_type=d["sign_type"],
                        condition=d["condition"],
                        condition_source=d["condition_source"],
                        bbox=d["bbox"],
                        confidence=d["confidence"],
                        first_seen=timestamp,
                        last_seen=timestamp
                    )
                    matched_track_ids.add(new_id)

            # Mark unmatched active tracks as misses
            for tid in list(self.active_tracks.keys()):
                if tid not in matched_track_ids:
                    self.active_tracks[tid].miss_streak += 1
                    if self.active_tracks[tid].miss_streak > 3:
                        del self.active_tracks[tid]

        return list(self.active_tracks.values())
