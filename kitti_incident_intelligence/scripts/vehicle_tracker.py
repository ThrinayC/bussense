"""
Vehicle and VRU Tracker with Range-Aware 3D Kalman Filtering
Performs 2D ByteTrack association coupled with 3D stereo depth extraction and trajectory estimation.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from scripts.detector import Detection
from scripts.track_filter import RangeAwareKalmanFilter


class TrackState(Enum):
    ACTIVE = "ACTIVE"
    TEMPORARILY_MISSING = "TEMPORARILY_MISSING"
    EXITED_SCENE = "EXITED_SCENE"
    LOST = "LOST"


class VehicleIdentityResolver:
    """
    Interface for vehicle identity resolution (PRD §53).
    Architecture permits future appearance embedding / re-identification.
    """
    def resolve_identity(self, track_id: int, appearance_crop: Optional[np.ndarray] = None) -> str:
        return f"TRACK_{track_id}"


def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


class TrackedObject:
    """
    Persistent track for a vehicle or vulnerable road user.
    """

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        init_X: float,
        init_Z: float,
        frame_id: int,
        timestamp: float,
        dt: float = 0.1,
        fB: float = 384.3815,
        fx: float = 721.5377,
    ):
        self.track_id = track_id
        self.cls_name = detection.cls_name
        self.category = detection.category
        self.bbox = detection.bbox.copy()
        self.conf = detection.conf

        self.state = TrackState.ACTIVE
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.frame_id = frame_id
        self.timestamp = timestamp

        # Semantic class history & stability
        self.class_history: List[str] = [detection.cls_name]
        self.stable_class: str = detection.cls_name
        self.is_associated_rider: bool = False
        self.associated_bike_id: Optional[int] = None

        # ANPR Plate Fields (strict persistence, never display unreadable)
        self.plate_text: Optional[str] = None
        self.plate_confidence: float = 0.0
        self.plate_verified: bool = False
        self.plate_observations: List[Dict] = []
        self.plate_last_confirmed_ts: Optional[float] = None

        # Range-aware 3D Kalman Filter
        self.kf = RangeAwareKalmanFilter(
            init_X=init_X,
            init_Z=init_Z,
            dt=dt,
            fB=fB,
            fx=fx,
        )

        # Rolling history (last 50 frames)
        self.history: List[Dict] = []
        self._record_history(init_X, init_Z, accepted=True)

    def _record_history(self, raw_X: float, raw_Z: float, accepted: bool):
        kf_state = self.kf.get_state()
        entry = {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "bbox": self.bbox.tolist(),
            "conf": self.conf,
            "raw_X": float(raw_X),
            "raw_Z": float(raw_Z),
            "X": kf_state["X"],
            "Z": kf_state["Z"],
            "Vx": kf_state["Vx"],
            "Vz": kf_state["Vz"],
            "sigma_Z": kf_state["sigma_Z"],
            "sigma_Vz": kf_state["sigma_Vz"],
            "accepted": accepted,
        }
        self.history.append(entry)
        if len(self.history) > 50:
            self.history.pop(0)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.state = TrackState.TEMPORARILY_MISSING

    def update(
        self,
        detection: Detection,
        raw_X: float,
        raw_Z: float,
        frame_id: int,
        timestamp: float,
        confidence: float = 1.0,
    ):
        self.bbox = detection.bbox.copy()
        self.conf = detection.conf
        self.cls_name = detection.cls_name
        self.class_history.append(detection.cls_name)
        if len(self.class_history) > 20:
            self.class_history.pop(0)

        # Determine stable class via frequency over rolling history
        from collections import Counter
        counts = Counter(self.class_history)
        self.stable_class = counts.most_common(1)[0][0]

        self.frame_id = frame_id
        self.timestamp = timestamp
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.ACTIVE

        # Update Kalman Filter
        _, accepted = self.kf.update(raw_X, raw_Z, confidence=confidence)
        self._record_history(raw_X, raw_Z, accepted=accepted)

    def update_plate_observation(self, raw_text: Optional[str], conf: float, timestamp: float):
        """
        Updates multi-frame ANPR observations with temporal consensus and persistence.
        """
        if not raw_text or conf < 0.30:
            return

        if self.category != "vehicle":
            return

        import re
        norm_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        if len(norm_text) < 4 or len(norm_text) > 10:
            return

        obs = {
            "text": norm_text,
            "conf": float(conf),
            "timestamp": timestamp,
            "frame_id": self.frame_id,
        }
        self.plate_observations.append(obs)
        if len(self.plate_observations) > 30:
            self.plate_observations.pop(0)

        # Temporal Consensus:
        # High confidence single reading (>= 0.80) or >= 2 consistent readings
        text_counts = {}
        for o in self.plate_observations:
            t = o["text"]
            text_counts[t] = text_counts.get(t, 0) + 1

        best_candidate, best_count = max(text_counts.items(), key=lambda x: x[1])
        cand_confs = [o["conf"] for o in self.plate_observations if o["text"] == best_candidate]
        avg_conf = sum(cand_confs) / len(cand_confs)

        if (best_count >= 2 and avg_conf >= 0.40) or (conf >= 0.80):
            self.plate_text = best_candidate
            self.plate_confidence = max(self.plate_confidence, avg_conf)
            self.plate_verified = True
            self.plate_last_confirmed_ts = timestamp

    @property
    def current_position_3d(self) -> Tuple[float, float]:
        """Returns (lateral X, longitudinal Z) in meters."""
        return self.kf.x[0], self.kf.x[1]

    @property
    def current_velocity_3d(self) -> Tuple[float, float]:
        """Returns (lateral Vx, longitudinal Vz) in m/s relative to camera."""
        return self.kf.x[2], self.kf.x[3]


class SceneTracker:
    """
    Multi-object tracker with 3D stereo association and ByteTrack lifecycle management.
    """

    def __init__(
        self,
        calib,
        dt: float = 0.1,
        max_lost_frames: int = 8,
        high_conf_thresh: float = 0.50,
        low_conf_thresh: float = 0.25,
        iou_threshold: float = 0.30,
    ):
        self.calib = calib
        self.dt = dt
        self.max_lost_frames = max_lost_frames
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.iou_threshold = iou_threshold

        self.next_track_id = 1
        self.tracks: List[TrackedObject] = []
        self.identity_resolver = VehicleIdentityResolver()

    def _extract_3d_position(
        self, bbox: np.ndarray, depth_map: np.ndarray, category: str
    ) -> Tuple[float, float, float, float]:
        """
        Extracts robust 3D coordinates from the stereo depth map within the target's bounding box.
        Returns: (X, Y, Z, confidence)
        """
        H, W = depth_map.shape
        x1, y1, x2, y2 = bbox
        x1_i, x2_i = int(max(0, x1)), int(min(W - 1, x2))
        y1_i, y2_i = int(max(0, y1)), int(min(H - 1, y2))

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0, 0.0, 0.0, 0.0

        box_w = x2_i - x1_i
        box_h = y2_i - y1_i

        # Sample target depth:
        # Vehicles: bottom 40% of height, middle 60% of width (ground/bumper contact cue)
        # VRUs: middle 60% of width, middle 60% of height (torso)
        if category == "vehicle":
            roi_y1 = int(y1_i + 0.55 * box_h)
            roi_y2 = int(y1_i + 0.95 * box_h)
        else:
            roi_y1 = int(y1_i + 0.20 * box_h)
            roi_y2 = int(y1_i + 0.80 * box_h)

        roi_x1 = int(x1_i + 0.20 * box_w)
        roi_x2 = int(x1_i + 0.80 * box_w)

        roi_depths = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
        valid_depths = roi_depths[(roi_depths > 1.0) & (roi_depths < 80.0)]

        if len(valid_depths) < 5:
            # Fallback to whole bbox
            whole_roi = depth_map[y1_i:y2_i, x1_i:x2_i]
            valid_depths = whole_roi[(whole_roi > 1.0) & (whole_roi < 80.0)]

        if len(valid_depths) < 5:
            return 0.0, 0.0, 0.0, 0.0

        # Robust median depth to reject outlier noise
        Z = float(np.median(valid_depths))
        coverage = len(valid_depths) / float(max(1, (roi_y2 - roi_y1) * (roi_x2 - roi_x1)))
        conf = float(np.clip(coverage * 1.5, 0.3, 1.0))

        # Horizontal center
        u_center = 0.5 * (x1 + x2)
        v_bottom = y2

        X = float((u_center - self.calib.cx) * Z / self.calib.fx)
        Y = float((v_bottom - self.calib.cy) * Z / self.calib.fy)

        return X, Y, Z, conf

    def update(
        self, detections: List[Detection], depth_map: np.ndarray, frame_id: int, timestamp: float
    ) -> List[TrackedObject]:
        """
        Updates tracks using ByteTrack two-stage association + stereo 3D localization.
        """
        # 1. Predict all existing tracks
        for t in self.tracks:
            t.predict()

        # 2. Split detections into high and low confidence
        high_dets = [d for d in detections if d.conf >= self.high_conf_thresh]
        low_dets = [d for d in detections if self.low_conf_thresh <= d.conf < self.high_conf_thresh]

        # 3. First Association: High-confidence detections with existing tracks
        matched_tracks, unmatched_tracks, unmatched_high_dets = self._associate_iou(
            self.tracks, high_dets, self.iou_threshold
        )

        for track_idx, det_idx in matched_tracks:
            det = high_dets[det_idx]
            X, Y, Z, conf = self._extract_3d_position(det.bbox, depth_map, det.category)
            if Z > 0.5:
                self.tracks[track_idx].update(det, X, Z, frame_id, timestamp, confidence=conf)

        # 4. Second Association: Low-confidence detections with remaining unmatched tracks
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        matched_low, unmatched_remaining, _ = self._associate_iou(
            remaining_tracks, low_dets, self.iou_threshold * 0.7
        )

        for trk_sub_idx, det_idx in matched_low:
            det = low_dets[det_idx]
            trk = remaining_tracks[trk_sub_idx]
            X, Y, Z, conf = self._extract_3d_position(det.bbox, depth_map, det.category)
            if Z > 0.5:
                trk.update(det, X, Z, frame_id, timestamp, confidence=conf * 0.7)

        # 5. Initialize new tracks from unmatched high-confidence detections
        for det_idx in unmatched_high_dets:
            det = high_dets[det_idx]
            X, Y, Z, conf = self._extract_3d_position(det.bbox, depth_map, det.category)
            if Z > 1.0:  # Only spawn track if valid 3D is observed
                new_track = TrackedObject(
                    track_id=self.next_track_id,
                    detection=det,
                    init_X=X,
                    init_Z=Z,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    dt=self.dt,
                    fB=self.calib.fB,
                    fx=self.calib.fx,
                )
                self.next_track_id += 1
                self.tracks.append(new_track)

        # 6. Lifecycle Management & Pruning
        active_tracks = []
        for t in self.tracks:
            # Check if track has moved outside camera boundaries
            x1, y1, x2, y2 = t.bbox
            if x2 < 10 or x1 > (self.calib.image_width - 10) or y2 < 10:
                t.state = TrackState.EXITED_SCENE

            if t.time_since_update > self.max_lost_frames:
                t.state = TrackState.LOST

            if t.state != TrackState.LOST:
                active_tracks.append(t)

        self.tracks = active_tracks
        return [t for t in self.tracks if t.state == TrackState.ACTIVE or t.time_since_update <= 2]

    def _associate_iou(
        self, tracks: List[TrackedObject], detections: List[Detection], threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU association between tracks and detections."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                # Only match same general class group
                if t.category == d.category:
                    iou_matrix[i, j] = compute_iou(t.bbox, d.bbox)

        matched: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        while True:
            max_iou = 0.0
            best_i, best_j = -1, -1
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_i, best_j = i, j

            if max_iou >= threshold:
                matched.append((best_i, best_j))
                unmatched_tracks.remove(best_i)
                unmatched_dets.remove(best_j)
            else:
                break

        return matched, list(unmatched_tracks), list(unmatched_dets)
