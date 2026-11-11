#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extreme Driving Tracking Diagnostic Tool
========================================
Diagnoses tracking vs detection localization on Extreme Driving sequences:
- Logs frame index, timestamp, image dimensions, detector bbox, tracker bbox, track ID, conf, center, dimensions, displacement
- Renders visual comparison overlay:
    DETECTION BOX: Green dashed/double border
    TRACK BOX: Blue/Orange solid box with ID and state
"""

import os
import sys
import json
import math
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kitti_incident_intelligence"))

from scripts.detector import ObjectDetector, Detection

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / float(areaA + areaB - inter)

class SimpleKalmanBoxTracker:
    """Standard 2D bounding box Kalman Tracker [cx, cy, w, h, vx, vy, vw, vh]."""
    def __init__(self, bbox, track_id, conf=1.0):
        self.track_id = track_id
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 50.0  # High initial velocity uncertainty
        
        self.conf = conf
        self.hits = 1
        self.time_since_update = 0
        self.state = "DETECTED"  # DETECTED, TRACKED, TEMPORARILY_OCCLUDED, LOST
        self.history = [(cx, cy)]
        
    def predict(self, dt=0.1):
        # State transition matrix F
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt
        
        # Process noise Q
        Q = np.eye(8, dtype=np.float32) * 1.0
        Q[4:, 4:] *= 4.0
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.state = "TEMPORARILY_OCCLUDED"
        if self.time_since_update > 2:
            self.state = "LOST"
            
    def update(self, bbox, conf, dt=0.1):
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        z = np.array([cx, cy, w, h], dtype=np.float32)
        
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        
        R = np.eye(4, dtype=np.float32) * 4.0  # Measurement noise
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ H) @ self.P
        
        self.hits += 1
        self.time_since_update = 0
        self.conf = conf
        self.state = "TRACKED"
        self.history.append((self.x[0], self.x[1]))
        if len(self.history) > 30:
            self.history.pop(0)
            
    def get_bbox(self):
        cx, cy, w, h = self.x[:4]
        return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float32)

def run_diagnostic():
    out_dir = ROOT / "outputs" / "evaluation" / "diagnostic_extreme_driving"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = ROOT / "potcrack_stereo" / "yolov8n.pt"
    if not weights_path.exists():
        weights_path = ROOT / "yolov8n.pt"
    detector = ObjectDetector(str(weights_path), conf_threshold=0.25)
    
    vid_path = ROOT / "kitti_incident_intelligence" / "data" / "extreme_driving" / "Critical_Driving.mp4"
    cap = cv2.VideoCapture(str(vid_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    dt = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"=== RUNNING EXTREME DRIVING TRACKING DIAGNOSTIC ===")
    print(f"Video: {vid_path.name}")
    print(f"FPS: {fps}, Total Frames: {total_frames}, dt: {dt:.4f}s")
    
    tracks = []
    next_id = 1
    log_records = []
    
    # Process 25 consecutive frames
    for f_idx in range(min(25, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        H, W = frame.shape[:2]
        # Extreme driving video has camera in left 640px and BEV map in right
        cam = frame[:, :640] if W >= 1000 else frame
        cam_h, cam_w = cam.shape[:2]
        ts = f_idx * dt
        
        # 1. Predict existing tracks
        for trk in tracks:
            trk.predict(dt=dt)
            
        # 2. Detect
        dets = detector.detect(cam)
        
        # 3. Associate via IoU
        matched_tracks = set()
        matched_dets = set()
        
        # Log pairs
        frame_logs = []
        
        for d_idx, d in enumerate(dets):
            best_iou = 0.0
            best_t_idx = -1
            for t_idx, trk in enumerate(tracks):
                if t_idx in matched_tracks:
                    continue
                iou = compute_iou(trk.get_bbox(), d.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_t_idx = t_idx
                    
            if best_iou >= 0.25:
                matched_tracks.add(best_t_idx)
                matched_dets.add(d_idx)
                trk = tracks[best_t_idx]
                prev_cx, prev_cy = trk.x[0], trk.x[1]
                trk.update(d.bbox, d.conf, dt=dt)
                
                disp = float(math.hypot(trk.x[0] - prev_cx, trk.x[1] - prev_cy))
                frame_logs.append({
                    "source_frame_index": int(f_idx),
                    "timestamp": round(float(ts), 3),
                    "image_dimensions": [int(cam_w), int(cam_h)],
                    "detector_bbox": [round(float(v), 1) for v in d.bbox],
                    "tracker_bbox": [round(float(v), 1) for v in trk.get_bbox()],
                    "track_id": int(trk.track_id),
                    "detection_confidence": round(float(d.conf), 3),
                    "bbox_center": [round(float(trk.x[0]), 1), round(float(trk.x[1]), 1)],
                    "bbox_dimensions": [round(float(trk.x[2]), 1), round(float(trk.x[3]), 1)],
                    "frame_to_frame_displacement": round(disp, 2),
                    "iou_match": round(float(best_iou), 3),
                    "state": str(trk.state),
                })
            elif d.conf >= 0.40:
                # Spawn new track
                new_trk = SimpleKalmanBoxTracker(d.bbox, next_id, d.conf)
                next_id += 1
                tracks.append(new_trk)
                frame_logs.append({
                    "source_frame_index": int(f_idx),
                    "timestamp": round(float(ts), 3),
                    "image_dimensions": [int(cam_w), int(cam_h)],
                    "detector_bbox": [round(float(v), 1) for v in d.bbox],
                    "tracker_bbox": [round(float(v), 1) for v in new_trk.get_bbox()],
                    "track_id": int(new_trk.track_id),
                    "detection_confidence": round(float(d.conf), 3),
                    "bbox_center": [round(float(new_trk.x[0]), 1), round(float(new_trk.x[1]), 1)],
                    "bbox_dimensions": [round(float(new_trk.x[2]), 1), round(float(new_trk.x[3]), 1)],
                    "frame_to_frame_displacement": 0.0,
                    "iou_match": 0.0,
                    "state": "DETECTED_NEW",
                })
                
        # Prune dead tracks
        tracks = [t for t in tracks if t.time_since_update <= 10]
        log_records.extend(frame_logs)
        
        # 4. Render visual diagnostic overlay:
        # DETECTION BOX = Green dashed style
        # TRACK BOX = Blue/Orange solid style with Track ID
        vis = cam.copy()
        
        # Draw detections in bright green
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(vis, f"DET: {d.cls_name} {d.conf:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                        
        # Draw tracks in blue / orange with ID
        for trk in tracks:
            if trk.time_since_update <= 1:  # Only render current or short prediction
                tb = trk.get_bbox()
                tx1, ty1, tx2, ty2 = [int(v) for v in tb]
                col = (255, 140, 0) if trk.time_since_update == 0 else (0, 140, 255)
                cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), col, 2)
                cv2.putText(vis, f"TRK #{trk.track_id} [{trk.state}]", (tx1, min(cam_h - 6, ty2 + 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
                            
        # Save key sample frames (0, 5, 10, 15, 20)
        if f_idx in [6, 7, 8, 9, 10, 15, 20]:
            cv2.imwrite(str(out_dir / f"diag_frame_{f_idx:02d}.jpg"), vis)
            
    cap.release()
    
    # Save log file
    log_path = out_dir / "extreme_tracking_diagnostic_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, indent=2)
        
    print(f"Diagnostic completed! Saved {len(log_records)} records to {log_path}")
    print(f"Saved diagnostic visualization frames to {out_dir}")

if __name__ == "__main__":
    run_diagnostic()
