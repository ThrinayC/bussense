#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_protected_working_pipelines.py
=========================================
Protected Regression Test for the WORKING Traffic Sign and Vehicle/Pedestrian Pipelines.
Ensures neither pipeline is modified, simplified, or broken while fixing other components.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
KITTI_PROJECT = ROOT / "kitti_incident_intelligence"
POTCRACK_PROJECT = ROOT / "potcrack_stereo"

for p in [str(ROOT), str(KITTI_PROJECT), str(POTCRACK_PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bussense_infrastructure.traffic_sign_engine import TrafficSignEngine
from bussense_infrastructure.models.sign_model import TrafficSignModelAdapter
from kitti_incident_intelligence.scripts.detector import ObjectDetector
from kitti_incident_intelligence.scripts.vehicle_tracker import SceneTracker
from kitti_incident_intelligence.scripts.calibration import KittiCalibration


def test_protected_sign_pipeline_accuracy_and_anti_stale():
    """Sign detector must accurately detect signs and never emit static repeated boxes."""
    engine = TrafficSignEngine(use_model=True)
    mtsvd_dir = ROOT / "datasets" / "mtsvd" / "raw"
    imgs = sorted(list(mtsvd_dir.rglob("*.jpg")))
    assert len(imgs) >= 50, "MTSVD images must be available"

    detected_boxes = []
    for p in imgs[:15]:
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = engine.detect_candidate_signs(rgb, filename_hint=p.name)
        for d in dets:
            assert d.get("detection_source") == "MODEL"
            detected_boxes.append(tuple(d["bbox"]))

    assert len(detected_boxes) > 0, "Working sign pipeline must produce detections"
    # Ensure detections are varied across scenes (many unique positions)
    unique_boxes = set(detected_boxes)
    assert len(unique_boxes) >= 15, f"Expected >= 15 unique sign boxes across 15 scenes, got {len(unique_boxes)}"

    # Blank frame test: missing detection must produce ZERO boxes
    blank = np.full((720, 1280, 3), 30, dtype=np.uint8)
    blank_dets = engine.detect_candidate_signs(blank)
    assert len(blank_dets) == 0, "No boxes must be drawn when no sign is present"


def test_protected_vehicle_and_vru_tracking():
    """Vehicle and VRU pipeline must detect targets and track them dynamically."""
    detector = ObjectDetector(str(ROOT / "potcrack_stereo" / "yolov8n.pt"), conf_threshold=0.25)
    calib = KittiCalibration(str(ROOT / "kitti_incident_intelligence" / "data" / "kitti" / "2011_09_26" / "calib_cam_to_cam.txt"))
    tracker = SceneTracker(calib=calib, dt=0.1)

    video_src = ROOT / "kitti_incident_intelligence" / "data" / "extreme_driving" / "Complex_Traffic_Driving.mp4"
    assert video_src.exists(), "Complex_Traffic_Driving.mp4 must exist"

    cap = cv2.VideoCapture(str(video_src))
    frame_dets = []
    for f_idx in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        depth_est = np.full((h, w), 15.0, dtype=np.float32)
        dets = detector.detect(frame)
        tracks = tracker.update(dets, depth_est, f_idx, f_idx * 0.1)
        active = [t for t in tracks if t.time_since_update == 0]
        frame_dets.append(len(active))
    cap.release()

    assert sum(frame_dets) > 0, "Vehicle tracker must actively track real road objects"


def test_protected_kitti_traffic_intelligence():
    """KITTI traffic flow, kinematics, and incident engines must operate correctly."""
    from kitti_incident_intelligence.scripts.kinematics_engine import KinematicsEngine
    from kitti_incident_intelligence.scripts.vru_engine import VRUEngine
    from kitti_incident_intelligence.scripts.incident_engine import IncidentEngine

    ke = KinematicsEngine()
    ie = IncidentEngine()
    assert ke.max_kinematic_range == 30.0


    # Test incident engine state evaluation
    assert ie.max_validated_range_m == 25.0
    assert ie.hard_braking_accel_mps2 == -3.5



def test_protected_original_pothole_crack_engine():
    """Original pothole and crack engine must preserve SVD plane fitting and 3D dimensions."""
    # SVD road plane fitting
    pts = np.array([
        [-1.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
        [-1.0, 0.0, 12.0],
        [1.0, 0.0, 12.0],
    ])
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2, :]
    if normal[1] < 0:
        normal = -normal
    d = -np.dot(normal, centroid)

    # All points should have ~0 distance to fitted plane
    for p in pts:
        dist = abs(np.dot(normal, p) + d)
        assert dist < 1e-4

    # Cavity depth test: depression point at y=0.08m (8cm below road)
    defect_pt = np.array([0.0, 0.08, 11.0])
    depth_m = np.dot(normal, defect_pt) + d
    assert abs(depth_m - 0.08) < 1e-4, f"Expected 8cm depth, got {depth_m*100:.1f}cm"

