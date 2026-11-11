#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for UnifiedBusSensePipeline and UnifiedFrameState.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_unified_pipeline import UnifiedBusSensePipeline, UnifiedFrameState


class TestUnifiedPipeline(unittest.TestCase):

    def setUp(self):
        self.calib = {
            "fx": 721.54,
            "fy": 721.54,
            "cx": 609.56,
            "cy": 172.85,
            "baseline": 0.4706
        }
        self.pipeline = UnifiedBusSensePipeline(calib=self.calib)

    def test_unified_frame_state_structure(self):
        """Verifies UnifiedFrameState has all mandatory fields."""
        H, W = 375, 1242
        # Synthetic stereo images
        left = np.zeros((H, W, 3), dtype=np.uint8)
        right = np.zeros((H, W, 3), dtype=np.uint8)

        # Draw a synthetic ground road in lower half
        left[200:, :] = 120
        right[200:, :] = 120

        state = self.pipeline.process_frame(left, right, frame_index=1, timestamp=0.1)

        self.assertIsInstance(state, UnifiedFrameState)
        self.assertEqual(state.frame_id, 1)
        self.assertAlmostEqual(state.timestamp, 0.1)
        self.assertIsInstance(state.vehicles, list)
        self.assertIsInstance(state.potholes, list)
        self.assertIsInstance(state.traffic_metrics, dict)
        self.assertIsInstance(state.road_plane, dict)
        self.assertIsInstance(state.stereo_quality, dict)
        self.assertIsInstance(state.performance_metrics, dict)

        # Mandatory traffic metric keys
        for k in ["visible_vehicle_count", "tracked_vehicle_count", "density_status"]:
            self.assertIn(k, state.traffic_metrics)

        # Mandatory stereo quality keys
        for k in ["valid_disp_fraction", "valid_depth_fraction", "status"]:
            self.assertIn(k, state.stereo_quality)

        # Mandatory performance metric keys
        for k in ["stereo_ms", "total_latency_ms", "effective_fps"]:
            self.assertIn(k, state.performance_metrics)

        # Backward compatibility dictionary access
        self.assertEqual(state["frame_index"], 1)
        self.assertEqual(state["vehicles"], state.vehicles)
        self.assertEqual(state["potholes"], state.potholes)
        self.assertIn("traffic", state)
        self.assertIn("density", state)
        self.assertIsNotNone(state.get("disparity"))
        self.assertIsNotNone(state.get("depth_m"))


if __name__ == "__main__":
    unittest.main()
