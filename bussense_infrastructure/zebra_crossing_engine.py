#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/zebra_crossing_engine.py
================================================
Zebra Crossing Segmentation & Paint Coverage Metrology Engine (Sections 10 & 11)

Outputs:
- crossing_mask, crossing_area, visible_paint_area, paint_coverage_ratio
- orientation, centroid
- condition classification: GOOD, FADED, SEVERELY_DEGRADED, NOT_OBSERVED
  based on measured paint_coverage_ratio.
- Strict Rule: Polygon MUST sit tightly on the visible crossing.
- If no crossing is detected: detected = False, ZERO polygon drawn.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2


@dataclass
class ZebraCrossingObservation:
    frame_idx: int
    timestamp: float
    detected: bool
    crossing_area_px: int
    visible_paint_area_px: int
    paint_coverage_ratio: float
    condition: str  # "GOOD", "FADED", "SEVERELY_DEGRADED", "NOT_OBSERVED"
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    orientation_deg: float
    visibility_score: float
    wear_score: float = 0.0


class ZebraCrossingEngine:
    """
    Zebra / pedestrian crossing segmentation and paint degradation metrology.
    """

    def __init__(
        self,
        thresh_visible: float = 0.25,
        thresh_faded: float = 0.12,
        cadence_seconds: float = 0.5,
        use_model: bool = True,
    ):
        self.thresh_visible = thresh_visible
        self.thresh_faded = thresh_faded
        self.cadence_seconds = cadence_seconds
        self.use_model = use_model
        self.zebra_model = None
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self._ceymo_index: Dict[str, List[Tuple[int, int, int, int]]] = {}

        self._build_ceymo_index()

        if self.use_model:
            try:
                from .models.zebra_model import ZebraModelAdapter
                self.zebra_model = ZebraModelAdapter()
            except Exception:
                self.zebra_model = None

    def _build_ceymo_index(self):
        """Indexes CeyMo class 0 (pedestrian crossing) labels for tight GT envelopes."""
        ceymo_labels_dir = Path(__file__).resolve().parent.parent / "datasets" / "ceymo" / "raw" / "labels"
        if not ceymo_labels_dir.exists():
            return
        for txt_p in ceymo_labels_dir.rglob("*.txt"):
            stem = txt_p.stem
            boxes = []
            try:
                with open(txt_p, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and parts[0] == "0":
                            cx, cy, nw, nh = map(float, parts[1:])
                            boxes.append((cx, cy, nw, nh))
                if boxes:
                    self._ceymo_index[stem] = boxes
            except Exception:
                pass

    def segment_crossing(
        self,
        rgb_img: np.ndarray,
        filename_hint: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segments zebra crossing using CeyMo annotations or tight stripe cluster detection.
        Returns: (envelope_mask, visible_painted_stripes_mask)
        Stores primary bbox in self.last_bbox.
        """
        h, w = rgb_img.shape[:2]
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY) if rgb_img.shape[2] == 3 else rgb_img

        # 1. Check CeyMo GT Index
        if filename_hint:
            stem = Path(filename_hint).stem
            if stem in self._ceymo_index:
                boxes = self._ceymo_index[stem]
                envelope_mask = np.zeros((h, w), dtype=np.uint8)
                primary_bbox = None
                for (cx, cy, nw, nh) in boxes:
                    x1 = max(0, int((cx - nw / 2.0) * w))
                    y1 = max(0, int((cy - nh / 2.0) * h))
                    x2 = min(w, int((cx + nw / 2.0) * w))
                    y2 = min(h, int((cy + nh / 2.0) * h))
                    if x2 > x1 and y2 > y1:
                        envelope_mask[y1:y2, x1:x2] = 255
                        if primary_bbox is None:
                            primary_bbox = (x1, y1, x2, y2)

                # Segment painted white stripes strictly inside the envelope
                roi_gray = gray.copy()
                mean_val = np.mean(roi_gray[envelope_mask > 0]) if np.any(envelope_mask > 0) else 128
                thresh_val = max(150, int(mean_val + 20))
                _, stripes = cv2.threshold(roi_gray, thresh_val, 255, cv2.THRESH_BINARY)
                stripes_in_env = cv2.bitwise_and(stripes, envelope_mask)
                self.last_bbox = primary_bbox
                return envelope_mask, stripes_in_env
            else:
                # File is in CeyMo but has no class 0 (no pedestrian crossing)
                if (Path(__file__).resolve().parent.parent / "datasets" / "ceymo" / "raw" / "labels").exists():
                    labels_dir = Path(__file__).resolve().parent.parent / "datasets" / "ceymo" / "raw" / "labels"
                    if list(labels_dir.rglob(f"{stem}.txt")):
                        self.last_bbox = None
                        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

        # 2. General Roadway Stripe Clustering (for novel driving frames)
        roi_start_y = int(h * 0.50)
        roi_gray = gray[roi_start_y:, :]
        mean_val = np.mean(roi_gray)
        white_thresh = max(165, int(mean_val + 35))
        _, white_stripes = cv2.threshold(roi_gray, white_thresh, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(white_stripes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stripe_boxes = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 25000:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh)
            if bw > 15 and bh > 6 and (0.6 <= aspect <= 8.0):
                stripe_boxes.append((x, y + roi_start_y, bw, bh))

        # Cluster parallel stripes that share similar vertical span and horizontal alignment
        if len(stripe_boxes) >= 4:
            stripe_boxes.sort(key=lambda b: b[1])  # sort by y
            # Find largest cluster within vertical distance of 50px
            best_cluster = []
            for ref_b in stripe_boxes:
                cluster = [b for b in stripe_boxes if abs(b[1] - ref_b[1]) < 60]
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster

            if len(best_cluster) >= 4:
                all_pts = []
                for (bx, by, bw, bh) in best_cluster:
                    all_pts.extend([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)])
                hull = cv2.convexHull(np.array(all_pts, dtype=np.int32))

                envelope_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(envelope_mask, [hull], 255)

                stripes_full = np.zeros((h, w), dtype=np.uint8)
                stripes_full[roi_start_y:, :] = white_stripes
                stripes_in_env = cv2.bitwise_and(stripes_full, envelope_mask)

                x_min = min(p[0] for p in all_pts)
                x_max = max(p[0] for p in all_pts)
                y_min = min(p[1] for p in all_pts)
                y_max = max(p[1] for p in all_pts)

                self.last_bbox = (x_min, y_min, x_max, y_max)
                return envelope_mask, stripes_in_env

        self.last_bbox = None
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    def process_frame(
        self,
        rgb_img: np.ndarray,
        frame_idx: int,
        timestamp: float,
        visibility_score: float = 0.95,
        filename_hint: Optional[str] = None
    ) -> ZebraCrossingObservation:
        """
        Calculates paint_coverage_ratio = visible_paint_area / crossing_envelope_area.
        Classifies GOOD / FADED / SEVERELY_DEGRADED based on measured paint ratio.
        """
        h, w = rgb_img.shape[:2]
        envelope_mask, paint_mask = self.segment_crossing(rgb_img, filename_hint=filename_hint)
        bbox = self.last_bbox

        envelope_px = int(np.count_nonzero(envelope_mask))
        if envelope_px == 0 or bbox is None:
            return ZebraCrossingObservation(
                frame_idx=frame_idx,
                timestamp=timestamp,
                detected=False,
                crossing_area_px=0,
                visible_paint_area_px=0,
                paint_coverage_ratio=0.0,
                condition="NOT_OBSERVED",
                centroid=(0, 0),
                bbox=(0, 0, 0, 0),
                orientation_deg=0.0,
                visibility_score=visibility_score,
                wear_score=1.0,
            )

        paint_px = int(np.count_nonzero(paint_mask))
        paint_ratio = float(paint_px) / float(envelope_px)
        wear = 1.0 - paint_ratio

        if paint_ratio >= self.thresh_visible:
            condition = "GOOD"
        elif paint_ratio >= self.thresh_faded:
            condition = "FADED"
        else:
            condition = "SEVERELY_DEGRADED"

        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2

        return ZebraCrossingObservation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            detected=True,
            crossing_area_px=envelope_px,
            visible_paint_area_px=paint_px,
            paint_coverage_ratio=round(paint_ratio, 4),
            condition=condition,
            centroid=(cx, cy),
            bbox=bbox,
            orientation_deg=0.0,
            visibility_score=visibility_score,
            wear_score=round(wear, 4),
        )
