#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for BusSense Vehicular Perception & Traffic Sensing Module.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure potcrack_stereo is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from traffic.detector import TrafficDetector, TARGET_CLASSES
from traffic.tracker import TrafficTracker
from traffic.spatial_analyzer import TrafficSpatialAnalyzer
from traffic.density_engine import TrafficDensityEngine


class TestTrafficPipeline(unittest.TestCase):
    """Test suite for traffic perception components."""

    def setUp(self):
        # Realistic camera parameters (e.g., ZED 2i / KITTI)
        self.fx = 721.5
        self.fy = 721.5
        self.cx = 600.0
        self.cy = 200.0
        self.baseline = 0.54
        self.img_w = 1242
        self.img_h = 375

        self.spatial_analyzer = TrafficSpatialAnalyzer(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            baseline=self.baseline, min_valid_disparity=0.5
        )
        self.density_engine = TrafficDensityEngine(
            fx=self.fx, img_width=self.img_w, img_height=self.img_h,
            z_min=2.0, z_max=35.0, window_size=10, min_inlier_ratio=0.40
        )

    def test_spatial_analyzer_accurate_3d(self):
        """Verify accurate 3D coordinate recovery from known disparity."""
        H, W = self.img_h, self.img_w
        disparity = np.zeros((H, W), dtype=np.float32)

        # Create a synthetic vehicle box at [550, 150, 650, 250]
        # Bottom-center is (600, 250) -> u=600 (=cx), v=250
        # Set disparity in lower 25% of box [225:250, 570:630] to 19.48 px
        # Expected depth Z = (721.5 * 0.54) / 19.48 = 20.0 m
        expected_Z = 20.0
        known_disp = (self.fx * self.baseline) / expected_Z  # ~19.48 px
        disparity[225:250, 570:630] = known_disp

        bbox = [550, 150, 650, 250]
        res = self.spatial_analyzer.compute_vehicle_3d_position(bbox, disparity)

        self.assertTrue(res["valid"])
        self.assertEqual(res["status"], "OK")
        self.assertAlmostEqual(res["Z_m"], expected_Z, delta=0.2)
        # u=600 == cx -> X should be ~0.0
        self.assertAlmostEqual(res["X_m"], 0.0, delta=0.2)
        # Y = (250 - 200) * 20.0 / 721.5 = 1.38 m
        self.assertAlmostEqual(res["Y_m"], 1.39, delta=0.2)

    def test_spatial_analyzer_unreliable_fail_safe(self):
        """Verify that zero or invalid disparity triggers DISTANCE: UNRELIABLE."""
        H, W = self.img_h, self.img_w
        disparity = np.zeros((H, W), dtype=np.float32)  # All zeros (no stereo match)

        bbox = [550, 150, 650, 250]
        res = self.spatial_analyzer.compute_vehicle_3d_position(bbox, disparity)

        self.assertFalse(res["valid"])
        self.assertIsNone(res["distance_m"])
        self.assertEqual(res["status"], "DISTANCE: UNRELIABLE")

    def test_inter_vehicle_distances_and_ego(self):
        """Verify Euclidean 3D distance and ego-vehicle proximity."""
        H, W = self.img_h, self.img_w
        disparity = np.zeros((H, W), dtype=np.float32)

        # Vehicle 1 at Z=10m, center
        disp1 = (self.fx * self.baseline) / 10.0
        disparity[225:250, 570:630] = disp1

        # Vehicle 2 at Z=25m, center (same lane)
        disp2 = (self.fx * self.baseline) / 25.0
        disparity[100:125, 580:620] = disp2

        tracks = [
            {"track_id": 1, "display_id": "Vehicle #01", "class": "car", "bbox": [550, 150, 650, 250], "confidence": 0.9},
            {"track_id": 2, "display_id": "Vehicle #02", "class": "truck", "bbox": [570, 50, 630, 125], "confidence": 0.85},
        ]

        scene = self.spatial_analyzer.analyze_scene(tracks, disparity)

        # Inter-vehicle distance should be ~15.0m longitudinal gap
        self.assertEqual(len(scene["inter_vehicle_distances"]), 1)
        pair = scene["inter_vehicle_distances"][0]
        self.assertTrue(pair["same_lane"])
        self.assertIn("SAME-LANE HEURISTIC", pair["same_lane_rule"])
        self.assertAlmostEqual(pair["longitudinal_gap_m"], 15.0, delta=0.5)

        # Ego vehicle should detect Vehicle #01 as nearest (~10m)
        ego = scene["ego_metrics"]
        self.assertEqual(ego["nearest_vehicle_id"], "Vehicle #01")
        self.assertAlmostEqual(ego["nearest_vehicle_distance_m"], 10.0, delta=0.5)

    def test_ray_road_plane_intersection_positioning(self):
        """Verify Fix 2: Camera ray / road-plane intersection for vehicle ground contact."""
        H, W = self.img_h, self.img_w
        disparity = np.zeros((H, W), dtype=np.float32)

        # Target vehicle at Z=15.0m, centered
        disp = (self.fx * self.baseline) / 15.0
        disparity[225:250, 570:630] = disp

        # Define a flat road plane: Y = 1.38m -> 0*X - 1*Y + 0*Z + 1.38 = 0
        road_plane = np.array([0.0, -1.0, 0.0, 1.388], dtype=np.float32)

        bbox = [550, 150, 650, 250]  # bottom center v=250 -> (250-200)/721.5 = 0.0693. Y = 1.388 -> Z = 1.388/0.0693 ~ 20m
        res = self.spatial_analyzer.compute_vehicle_3d_position(bbox, disparity, road_plane=road_plane)

        self.assertTrue(res["valid"])
        self.assertIn(res["position_method"], ["ray_road_plane_intersection", "stereo_ground_roi"])

    def test_density_engine_computation_and_guardrails(self):
        """Verify Fix 3: Explicit 3D road plane polygon area calculation and UNRELIABLE guardrails."""
        # Case 1: Healthy road plane (90% inliers)
        healthy_plane = np.array([0.0, -1.0, 0.05, 1.65], dtype=np.float32)
        area, vertices, reliable, status = self.density_engine.compute_road_plane_polygon_area(
            healthy_plane, road_plane_inliers=900, total_road_pts=1000
        )
        self.assertTrue(reliable)
        self.assertIsNotNone(area)
        self.assertGreater(area, 20.0)
        self.assertEqual(len(vertices), 4)

        # Case 2: Degraded road plane (only 20% inliers < 40% threshold)
        area_bad, _, reliable_bad, status_bad = self.density_engine.compute_road_plane_polygon_area(
            healthy_plane, road_plane_inliers=200, total_road_pts=1000
        )
        self.assertFalse(reliable_bad)
        self.assertIn("UNRELIABLE", status_bad)

    def test_tracker_id_formatting(self):
        """Verify tracker formats persistent IDs as Vehicle #01, Vehicle #02."""
        tracker = TrafficTracker()
        id1 = tracker._format_id(42)
        id2 = tracker._format_id(99)
        id1_repeat = tracker._format_id(42)

        self.assertEqual(id1, "Vehicle #01")
        self.assertEqual(id2, "Vehicle #02")
        self.assertEqual(id1_repeat, "Vehicle #01")  # Must be persistent!


if __name__ == "__main__":
    unittest.main()
