#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Vehicular Perception: Traffic Tracker
Multi-object vehicle tracking with ByteTrack association.
Assigns formatted persistent IDs (e.g., 'Vehicle #01', 'Vehicle #02').
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
from .detector import TARGET_CLASSES, DEFAULT_WEIGHT_PATHS


class TrafficTracker:
    """Multi-object vehicle tracker ensuring ID persistence across consecutive frames."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        tracker_type: str = "bytetrack.yaml",
        device: Optional[str] = None
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_type = tracker_type
        self.device = device

        resolved_model = None
        if model_path and os.path.exists(model_path):
            resolved_model = model_path
        else:
            for p in DEFAULT_WEIGHT_PATHS:
                if p.exists():
                    resolved_model = str(p)
                    break

        if resolved_model is None:
            resolved_model = "yolov8n.pt"

        self.model = YOLO(resolved_model)
        if device:
            self.model.to(device)

        # Mapping from raw numeric track ID to sequential formatted ID
        self.id_map: Dict[int, str] = {}
        self.next_seq_num: int = 1

    def reset(self):
        """Reset tracking state for a new video or sequence."""
        self.id_map.clear()
        self.next_seq_num = 1

    def _format_id(self, raw_id: int) -> str:
        """Map raw tracker ID to formatted sequential ID: 'Vehicle #01'."""
        if raw_id not in self.id_map:
            self.id_map[raw_id] = f"Vehicle #{self.next_seq_num:02d}"
            self.next_seq_num += 1
        return self.id_map[raw_id]

    def track(self, rgb_img: np.ndarray, frame_index: int = 0) -> List[Dict[str, Any]]:
        """
        Track vehicles in frame with temporal persistence.

        Args:
            rgb_img: HxWx3 uint8 BGR/RGB image.
            frame_index: Current frame number.

        Returns:
            List of tracked objects:
            {
                'track_id': int,
                'display_id': str ('Vehicle #01'),
                'class': str ('car', 'bus', 'truck'),
                'confidence': float,
                'bbox': [x1, y1, x2, y2],
                'frame_index': int
            }
        """
        results = self.model.track(
            source=rgb_img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=list(TARGET_CLASSES.keys()),
            persist=True,
            tracker=self.tracker_type,
            verbose=False,
            device=self.device
        )

        tracks = []
        if not results or len(results) == 0:
            return tracks

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return tracks

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            raw_id = int(box.id[0].item()) if box.id is not None else -1

            display_id = self._format_id(raw_id) if raw_id >= 0 else "Vehicle #??"
            cls_name = TARGET_CLASSES.get(cls_id, "vehicle")

            tracks.append({
                "track_id": raw_id,
                "display_id": display_id,
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "frame_index": frame_index
            })

        return tracks
