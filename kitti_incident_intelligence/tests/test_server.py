"""
Automated Smoke Tests for Server Endpoints and Offline Queue — Phase 10
Tests FastAPI ingestion/retrieval, spatial aggregation, and offline queue resilience per PRD §49 & §52.
"""

import sys
import os
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.main import app
from scripts.evidence_buffer import LocalOfflineQueue


class TestServerAndQueue(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")

    def test_event_ingestion_and_retrieval(self):
        event_payload = {
            "event_id": "EVT_TEST_001",
            "event_type": "RAPID_CLOSING",
            "state": "UNVERIFIED",
            "timestamp": 12.4,
            "frame_id": 42,
            "gps": {"lat": 49.0145, "lon": 8.4324},
            "ego": {"speed_mps": 11.2, "yaw_rate": -0.01},
            "vehicles": [{"track_id": 3}],
            "vru": {"track_id": None, "class": None},
            "metrics": {
                "distance_m": 12.5,
                "relative_speed_mps": -4.2,
                "closing_rate_mps": 4.2,
                "time_headway_s": 1.12,
                "ttc_kinematic_s": 2.98,
                "ttc_looming_s": None,
            },
            "uncertainty": {"range_qualified": True},
            "registration": {"number": None, "confidence": None},
            "evidence": {"pre_event": [], "event": ["frame_0042.jpg"], "post_event": []},
        }

        # POST event
        post_resp = self.client.post("/api/events", json=event_payload)
        self.assertEqual(post_resp.status_code, 201)
        self.assertEqual(post_resp.json()["status"], "ACK")

        # GET event by ID
        get_resp = self.client.get("/api/events/EVT_TEST_001")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["event_id"], "EVT_TEST_001")
        self.assertEqual(get_resp.json()["state"], "UNVERIFIED")

    def test_vru_observations_and_area_aggregation(self):
        # Ingest multiple observations in same spatial segment
        for i in range(3):
            obs = {
                "observation_id": f"OBS_TEST_{i}",
                "timestamp": 10.0 + i,
                "gps": {"lat": 49.0140, "lon": 8.4320},
                "road_segment_id": "SEG_TEST_MAIN_ROAD",
                "vru_class": "person",
                "crossing": True,
                "in_path": True,
                "near_miss": False,
                "small_stature_candidate": False,
                "distance_m": 8.5,
                "ttc_s": 2.1,
            }
            resp = self.client.post("/api/vru-observations", json=obs)
            self.assertEqual(resp.status_code, 201)

        # GET aggregated vulnerable areas
        areas_resp = self.client.get("/api/vru-areas?min_observations=2")
        self.assertEqual(areas_resp.status_code, 200)
        data = areas_resp.json()
        self.assertGreaterEqual(data["total_vulnerable_segments"], 1)

    def test_offline_queue_retention_and_ack(self):
        queue = LocalOfflineQueue(queue_dir="outputs/evidence/test_queue")
        sample_event = {"event_id": "EVT_QUEUE_TEST", "event_type": "CUT_IN_CANDIDATE"}

        # Enqueue event
        qpath = queue.enqueue_event(sample_event)
        self.assertTrue(os.path.exists(qpath))

        # Check pending
        pending = queue.get_pending_events()
        self.assertTrue(any(p[1]["event_id"] == "EVT_QUEUE_TEST" for p in pending))

        # Mark uploaded
        queue.mark_uploaded(qpath)

        # Verify not pending
        pending_after = queue.get_pending_events()
        self.assertFalse(any(p[1]["event_id"] == "EVT_QUEUE_TEST" for p in pending_after))


if __name__ == "__main__":
    unittest.main()
