"""
BusSense — Vehicular Perception: Traffic Detector
Modular vehicle detection on LEFT RGB imagery using YOLO.

Supported Detector Classes (Standard MS-COCO):
- car (class 2)
- motorcycle (class 3)
- bus (class 5)
- truck (class 7)

NOTE on Class Handling:
- MS-COCO does NOT have a dedicated 'van' class. In COCO-trained YOLO models,
  vans are naturally detected under 'car' or 'truck'.
- For KITTI evaluation, KITTI's ground-truth 'Van' is mapped to generic vehicles
  along with 'Car' (i.e. 'car' maps to ['Car', 'Van']).
- Pedestrians (class 0) and cyclists (class 1) are explicitly excluded from vehicular metrics.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

# Target classes supported by standard COCO YOLO models
TARGET_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Explicit supported vehicle categories (NO phantom 'van' class in COCO)
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Search paths for default YOLO weights
DEFAULT_WEIGHT_PATHS = [
    Path("yolov8n.pt"),
    Path(__file__).resolve().parent.parent / "weights" / "yolov8n.pt",
    Path(r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\yolov8n.pt"),
]


class TrafficDetector:
    """Modular vehicle detector operating on RGB images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
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
            resolved_model = "yolov8n.pt"  # Ultralytics will auto-download or cache

        self.model = YOLO(resolved_model)
        if device:
            self.model.to(device)

    def detect(self, rgb_img: np.ndarray, frame_id: int = 0) -> List[Dict[str, Any]]:
        """
        Detect vehicles in a single RGB frame.

        Args:
            rgb_img: HxWx3 uint8 BGR/RGB image.
            frame_id: Integer sequence index.

        Returns:
            List of detection dicts:
            {
                'bbox': [x1, y1, x2, y2],
                'class': str ('car', 'bus', 'truck', 'motorcycle'),
                'confidence': float,
                'class_id': int,
                'frame_id': int
            }
        """
        results = self.model.predict(
            source=rgb_img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=list(TARGET_CLASSES.keys()),
            verbose=False,
            device=self.device
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            cls_name = TARGET_CLASSES.get(cls_id, "vehicle")

            detections.append({
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "class": cls_name,
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "frame_id": frame_id
            })

        return detections
