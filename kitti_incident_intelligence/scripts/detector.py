"""
Object Detector for Vehicles and Vulnerable Road Users (VRUs)
Wraps YOLOv8n with strict target class filtering (car, truck, bus, motorcycle, person, bicycle).
"""

from typing import List, Dict, NamedTuple
import numpy as np
from ultralytics import YOLO


class Detection(NamedTuple):
    bbox: np.ndarray  # [x1, y1, x2, y2]
    cls_name: str     # 'car', 'person', etc.
    conf: float       # [0, 1]
    category: str     # 'vehicle' or 'vru'


class ObjectDetector:
    """
    YOLO-based object detector filtering strictly for road safety targets.
    """

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
    VRU_CLASSES = {"person", "bicycle", "motorcycle"}  # Motorcycle bridges both traffic & vulnerable

    def __init__(self, weights_path: str, conf_threshold: float = 0.25):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(weights_path)
        self.class_names = self.model.names

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.model(image_bgr, conf=self.conf_threshold, verbose=False)[0]
        detections: List[Detection] = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.class_names.get(cls_id, "")
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()

            if cls_name in self.VRU_CLASSES:
                category = "vru"
            elif cls_name in self.VEHICLE_CLASSES:
                category = "vehicle"
            else:
                continue

            detections.append(
                Detection(
                    bbox=xyxy,
                    cls_name=cls_name,
                    conf=conf,
                    category=category,
                )
            )

        return detections
