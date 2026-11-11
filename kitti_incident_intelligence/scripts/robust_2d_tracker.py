#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust 2D ByteTrack & Kalman Tracker with Track Quality Enforcement
===================================================================
Conforms strictly to BusSense Track Quality & Modality Rules:
1. Explicit internal states:
   - DETECTED: Just initialized on high-confidence detection.
   - TRACKED: Firmly associated in current frame.
   - TEMPORARILY_OCCLUDED: Missed detection, short Kalman prediction (<= 2 frames).
   - REACQUIRING: Re-detected after occlusion.
   - LOST: time_since_update > 2 frames. Visual rendering SUPPRESSED.
2. Visual Rendering Rule:
   A box is allowed to exist visually ONLY if time_since_update <= 2 and
   state in [DETECTED, TRACKED, TEMPORARILY_OCCLUDED, REACQUIRING].
   No long unsupported extrapolations or drifting boxes.
3. Timestamp & Frame-Rate Aware:
   Uses true video dt = t_current - t_last for physical Kalman motion update.
4. Scale & Coordinate Mapping:
   Supports arbitrary inference resolution -> rendered frame coordinates mapping.
"""

import math
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

class TrackQualityState(str, Enum):
    DETECTED = "DETECTED"
    TRACKED = "TRACKED"
    TEMPORARILY_OCCLUDED = "TEMPORARILY_OCCLUDED"
    REACQUIRING = "REACQUIRING"
    LOST = "LOST"

def compute_2d_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return float(inter / (areaA + areaB - inter))

class Kalman2DBoxFilter:
    """Kalman filter tracking [cx, cy, w, h, vx, vy, vw, vh]."""
    def __init__(self, bbox: np.ndarray):
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 20.0
        self.P[4:, 4:] *= 40.0

    def predict(self, dt: float = 0.1):
        dt = max(0.01, min(0.5, dt))
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt
        Q = np.eye(8, dtype=np.float32) * 1.5
        Q[4:, 4:] *= 5.0
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, bbox: np.ndarray):
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        z = np.array([cx, cy, w, h], dtype=np.float32)
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        R = np.eye(4, dtype=np.float32) * 4.0
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ H) @ self.P

    def get_bbox(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w = max(2.0, w)
        h = max(2.0, h)
        return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float32)

class Tracked2DObject:
    """Individual 2D tracked object with quality states and lifecycle management."""
    def __init__(
        self,
        track_id: int,
        bbox: np.ndarray,
        cls_name: str,
        category: str,
        conf: float,
        timestamp: float,
        frame_idx: int,
    ):
        self.track_id = track_id
        self.cls_name = cls_name
        self.category = category
        self.conf = conf
        self.bbox = bbox.copy()
        self.state = TrackQualityState.DETECTED
        
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_timestamp = timestamp
        self.last_frame_idx = frame_idx
        
        self.kf = Kalman2DBoxFilter(bbox)
        
        # ANPR fields (strict HUD rule)
        self.plate_text: Optional[str] = None
        self.plate_confidence: float = 0.0
        self.plate_verified: bool = False
        self.vehicle_crops: List[np.ndarray] = []
        self.plate_crops: List[np.ndarray] = []
        
        # History
        self.history: List[Dict[str, Any]] = []
        self._record(frame_idx, timestamp)

    def _record(self, frame_idx: int, timestamp: float):
        b = self.get_bbox()
        cx = float(0.5 * (b[0] + b[2]))
        cy = float(0.5 * (b[1] + b[3]))
        w = float(max(1.0, b[2] - b[0]))
        h = float(max(1.0, b[3] - b[1]))
        aspect = float(h / w)
        speed = 0.0
        vx, vy = 0.0, 0.0
        if self.history:
            prev = self.history[-1]
            dt = max(1e-3, timestamp - prev["timestamp"])
            vx = (cx - prev["cx"]) / dt
            vy = (cy - prev["cy"]) / dt
            speed = float(math.hypot(vx, vy))
            
        self.history.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "bbox": [float(v) for v in b],
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aspect": aspect,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "speed_px_s": speed,
            "state": self.state.value,
        })
        if len(self.history) > 60:
            self.history.pop(0)

    def update_plate_observation(self, text: Optional[str], conf: float, plate_crop: Optional[np.ndarray] = None, min_consensus: int = 2):
        if not text or conf < 0.30:
            return
        import re
        norm = re.sub(r'[^A-Z0-9]', '', str(text).upper())
        if len(norm) < 4 or len(norm) > 10:
            return
        
        # Non-vehicle gating
        if self.category != "vehicle" and not any(v in self.cls_name.lower() for v in ["car", "truck", "bus", "van", "vehicle"]):
            return

        if not hasattr(self, "plate_observations"):
            self.plate_observations = []
        self.plate_observations.append({"text": norm, "conf": float(conf), "crop": plate_crop})
        if len(self.plate_observations) > 30:
            self.plate_observations.pop(0)

        # Check temporal consensus
        counts: Dict[str, int] = {}
        for obs in self.plate_observations:
            t = obs["text"]
            counts[t] = counts.get(t, 0) + 1

        best_cand, best_count = max(counts.items(), key=lambda x: x[1])
        confs = [obs["conf"] for obs in self.plate_observations if obs["text"] == best_cand]
        avg_conf = sum(confs) / len(confs)

        if (best_count >= min_consensus and avg_conf >= 0.50) or (conf >= 0.88):
            cl = re.sub(r'[^A-Z0-9]', '', best_cand)
            if len(cl) == 7 and cl[:2].isalpha() and cl[2:].isdigit():
                self.plate_text = f"{cl[:2]} {cl[2:]}"
            elif len(cl) == 6 and cl[:2].isalpha() and cl[2:].isdigit():
                self.plate_text = f"{cl[:2]} {cl[2:]}"
            elif len(cl) == 7 and cl[:3].isalpha() and cl[3:].isdigit():
                self.plate_text = f"{cl[:3]} {cl[3:]}"
            elif len(cl) == 8 and cl[:2].isalpha() and cl[2:5].isdigit() and cl[5:].isalpha():
                self.plate_text = f"{cl[:2]} {cl[2:5]} {cl[5:]}"
            else:
                self.plate_text = best_cand
            self.plate_confidence = max(self.plate_confidence, avg_conf)
            self.plate_verified = True
            if plate_crop is not None:
                self.plate_crops.append(plate_crop)

    def predict(self, timestamp: float):
        dt = timestamp - self.last_timestamp if timestamp > self.last_timestamp else 0.1
        self.kf.predict(dt)
        self.bbox = self.kf.get_bbox()
        self.age += 1

    def update(self, bbox: np.ndarray, conf: float, cls_name: str, timestamp: float, frame_idx: int):
        was_occluded = (self.time_since_update > 0 or self.state in [TrackQualityState.TEMPORARILY_OCCLUDED, TrackQualityState.LOST])
        self.kf.update(bbox)
        self.bbox = self.kf.get_bbox()
        self.conf = conf
        self.cls_name = cls_name
        self.hits += 1
        self.time_since_update = 0
        self.last_timestamp = timestamp
        self.last_frame_idx = frame_idx
        
        if was_occluded and self.hits > 1:
            self.state = TrackQualityState.REACQUIRING
        else:
            self.state = TrackQualityState.TRACKED
            
        self._record(frame_idx, timestamp)

    def mark_missed(self):
        self.time_since_update += 1
        if 1 <= self.time_since_update <= 2:
            self.state = TrackQualityState.TEMPORARILY_OCCLUDED
        else:
            self.state = TrackQualityState.LOST

    def get_bbox(self) -> np.ndarray:
        return self.bbox.copy()

    @property
    def is_renderable(self) -> bool:
        """Visual rendering gate: never render lost tracks or predictions older than 2 frames."""
        return (self.time_since_update <= 2 and self.state != TrackQualityState.LOST)

class Robust2DTracker:
    """
    Two-stage ByteTrack 2D engine with physical Kalman prediction and coordinate scaling.
    """
    def __init__(
        self,
        high_conf_thresh: float = 0.40,
        low_conf_thresh: float = 0.15,
        iou_match_thresh: float = 0.25,
        max_lost_frames: int = 20,
    ):
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.iou_match_thresh = iou_match_thresh
        self.max_lost_frames = max_lost_frames
        
        self.tracks: List[Tracked2DObject] = []
        self.next_track_id = 1
        self.last_timestamp = 0.0

    def reset(self):
        self.tracks = []
        self.next_track_id = 1
        self.last_timestamp = 0.0

    def update(
        self,
        detections: List[Any],
        timestamp: float,
        frame_idx: int,
        source_shape: Tuple[int, int],
        render_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Tracked2DObject]:
        """
        Updates 2D tracks with detection bounding boxes.
        source_shape: (height, width) of input fed to detector.
        render_shape: (height, width) of final canvas to map coordinates onto.
        """
        src_h, src_w = source_shape
        if render_shape is not None:
            rnd_h, rnd_w = render_shape
            sx = rnd_w / float(src_w)
            sy = rnd_h / float(src_h)
        else:
            sx, sy = 1.0, 1.0

        # Scale detection bboxes to render canvas coordinates
        scaled_dets = []
        for d in detections:
            if isinstance(d, dict):
                bx = np.array(d["bbox"], dtype=np.float32)
                c_name = d.get("cls_name", "car")
                cat = d.get("category", "vehicle")
                c_val = float(d.get("conf", 0.80))
            else:
                bx = np.array(d.bbox, dtype=np.float32)
                c_name = d.cls_name
                cat = d.category
                c_val = float(d.conf)

            scaled_bbox = np.array([bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy], dtype=np.float32)
            scaled_dets.append({
                "bbox": scaled_bbox,
                "cls_name": c_name,
                "category": cat,
                "conf": c_val,
            })

        # 1. Kalman Prediction for all existing tracks
        for trk in self.tracks:
            trk.predict(timestamp)

        # 2. Partition detections into high and low confidence
        high_dets = [d for d in scaled_dets if d["conf"] >= self.high_conf_thresh]
        low_dets = [d for d in scaled_dets if self.low_conf_thresh <= d["conf"] < self.high_conf_thresh]

        # 3. First Association: High-confidence detections with active/occluded tracks
        matched_tracks_1, unmatched_trks_1, unmatched_high_dets = self._match_iou(
            self.tracks, high_dets, self.iou_match_thresh
        )

        matched_trk_indices = set()
        for t_idx, d_idx in matched_tracks_1:
            det = high_dets[d_idx]
            self.tracks[t_idx].update(
                det["bbox"], det["conf"], det["cls_name"], timestamp, frame_idx
            )
            matched_trk_indices.add(t_idx)

        # 4. Second Association: Remaining tracks with low-confidence detections
        rem_trks_with_orig_idx = [(i, self.tracks[i]) for i in unmatched_trks_1]
        rem_trks = [item[1] for item in rem_trks_with_orig_idx]
        matched_tracks_2, unmatched_trks_2, _ = self._match_iou(
            rem_trks, low_dets, self.iou_match_thresh * 0.8
        )

        for sub_t_idx, d_idx in matched_tracks_2:
            orig_t_idx = rem_trks_with_orig_idx[sub_t_idx][0]
            det = low_dets[d_idx]
            self.tracks[orig_t_idx].update(
                det["bbox"], det["conf"] * 0.8, det["cls_name"], timestamp, frame_idx
            )
            matched_trk_indices.add(orig_t_idx)

        # 5. Mark remaining unmatched tracks as missed
        for i, trk in enumerate(self.tracks):
            if i not in matched_trk_indices:
                trk.mark_missed()

        # 6. Spawn new tracks from unmatched high-confidence detections
        for d_idx in unmatched_high_dets:
            det = high_dets[d_idx]
            new_trk = Tracked2DObject(
                track_id=self.next_track_id,
                bbox=det["bbox"],
                cls_name=det["cls_name"],
                category=det["category"],
                conf=det["conf"],
                timestamp=timestamp,
                frame_idx=frame_idx,
            )
            self.next_track_id += 1
            self.tracks.append(new_trk)

        # 7. Prune dead tracks that exceeded max_lost_frames
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]
        self.last_timestamp = timestamp

        # Return renderable tracks by default
        return [t for t in self.tracks if t.is_renderable]

    def get_all_tracks(self) -> List[Tracked2DObject]:
        """Returns all tracks (including occluded/lost) for internal state/event processing."""
        return self.tracks

    def get_renderable_tracks(self) -> List[Tracked2DObject]:
        """Returns only tracks meeting the strict track quality render rule."""
        return [t for t in self.tracks if t.is_renderable]

    def _match_iou(
        self, tracks: List[Tracked2DObject], dets: List[Dict[str, Any]], thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        iou_mat = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, t in enumerate(tracks):
            tb = t.get_bbox()
            for j, d in enumerate(dets):
                if t.category == d["category"]:
                    iou_mat[i, j] = compute_2d_iou(tb, d["bbox"])

        matched = []
        unmatched_trks = set(range(len(tracks)))
        unmatched_dets = set(range(len(dets)))

        while True:
            max_iou = 0.0
            best_i, best_j = -1, -1
            for i in unmatched_trks:
                for j in unmatched_dets:
                    if iou_mat[i, j] > max_iou:
                        max_iou = iou_mat[i, j]
                        best_i, best_j = i, j

            if max_iou >= thresh:
                matched.append((best_i, best_j))
                unmatched_trks.remove(best_i)
                unmatched_dets.remove(best_j)
            else:
                break

        return matched, list(unmatched_trks), list(unmatched_dets)
