"""
Evidence Buffer and Local Offline Queue — Phase 10
Maintains rolling image buffers, packages pre/event/post evidence, and manages
the offline event queue per PRD §46 & §52.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class EvidenceBuffer:
    """
    Maintains a rolling circular buffer of recent frames and generates evidence clips/crops.
    """

    def __init__(self, capacity: int = 30, output_dir: str = "outputs/evidence"):
        self.capacity = capacity
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_buffer: List[Dict] = []

    def push_frame(self, frame_id: int, image_bgr: np.ndarray, timestamp: float):
        self.frame_buffer.append({
            "frame_id": frame_id,
            "image": image_bgr.copy(),
            "timestamp": timestamp,
        })
        if len(self.frame_buffer) > self.capacity:
            self.frame_buffer.pop(0)

    def capture_event_evidence(
        self, event_pkg: Dict, target_bbox: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Saves pre-event, event, post-event frames and vehicle crop.
        Updates event_pkg['evidence'] paths.
        """
        event_id = event_pkg["event_id"]
        frame_id = event_pkg["frame_id"]

        event_dir = os.path.join(self.output_dir, event_id)
        os.makedirs(event_dir, exist_ok=True)

        # Save event frame
        if self.frame_buffer:
            curr_img = self.frame_buffer[-1]["image"]
            event_img_path = os.path.join(event_dir, f"event_{frame_id:04d}.jpg")
            cv2.imwrite(event_img_path, curr_img)
            event_pkg["evidence"]["event"].append(event_img_path)

            # Pre-event frame (from 5 frames ago if available)
            pre_idx = max(0, len(self.frame_buffer) - 6)
            pre_img = self.frame_buffer[pre_idx]["image"]
            pre_path = os.path.join(event_dir, f"pre_{self.frame_buffer[pre_idx]['frame_id']:04d}.jpg")
            cv2.imwrite(pre_path, pre_img)
            event_pkg["evidence"]["pre_event"].append(pre_path)

            # Target crop if bbox provided
            if target_bbox is not None:
                x1, y1, x2, y2 = target_bbox.astype(int)
                H, W, _ = curr_img.shape
                x1, x2 = max(0, x1), min(W - 1, x2)
                y1, y2 = max(0, y1), min(H - 1, y2)
                if x2 > x1 and y2 > y1:
                    crop = curr_img[y1:y2, x1:x2]
                    crop_path = os.path.join(event_dir, "target_crop.jpg")
                    cv2.imwrite(crop_path, crop)
                    event_pkg["evidence"]["vehicle_crop"] = crop_path

        # Save event metadata JSON
        json_path = os.path.join(event_dir, "event_package.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(event_pkg, f, indent=2)

        return event_pkg


class LocalOfflineQueue:
    """
    Local event queue ensuring reliable upload when connectivity is interrupted (PRD §52).
    """

    def __init__(self, queue_dir: str = "outputs/evidence/queue"):
        self.queue_dir = queue_dir
        os.makedirs(self.queue_dir, exist_ok=True)

    def enqueue_event(self, event_pkg: Dict) -> str:
        event_id = event_pkg["event_id"]
        path = os.path.join(self.queue_dir, f"{event_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"uploaded": False, "queued_at": time.time(), "event": event_pkg}, f, indent=2)
        return path

    def get_pending_events(self) -> List[Tuple[str, Dict]]:
        pending = []
        for fname in sorted(os.listdir(self.queue_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.queue_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("uploaded", False):
                pending.append((path, data["event"]))
        return pending

    def mark_uploaded(self, queue_path: str):
        with open(queue_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["uploaded"] = True
        data["uploaded_at"] = time.time()
        with open(queue_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
