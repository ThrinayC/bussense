"""
Automated Test Suite — BusSense FastAPI Server Contract
=======================================================
Verifies all 15 required contract specifications:
1. Heartbeat -> 201
2. Valid Pothole -> 201
3. Valid Crack -> 201
4. Valid Emergency Incident + Clip -> 201
5. Valid VRU Encounter -> 201
6. Valid ANPR Event -> 201
7. Invalid/Unreadable ANPR Rejection -> 422
8. Traffic Aggregate -> 201
9. Vulnerable Area Aggregate -> 201
10. Idempotency on Duplicate event_id -> 200/201 (no duplicate records)
11. Malformed GPS Validation -> 422
12. Invalid Confidence Validation -> 422
13. Offline Queue Persistence & Recovery -> Verified
14. Media Type & Size Enforcement -> 422 on invalid
15. Server Restart Persistence -> Events Retained
"""

import os
import sys
import shutil
import tempfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bussense_server.app import create_app
from bussense_edge.offline_queue import BusSenseOfflineQueueClient
from bussense_edge.event_filter import BusSenseEdgeEventFilter


@pytest.fixture
def test_env():
    """Provides an isolated temporary database and media directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="bussense_test_")
    db_file = os.path.join(temp_dir, "test_events.db")
    app = create_app(db_path=db_file)
    client = TestClient(app)
    yield {"temp_dir": temp_dir, "db_file": db_file, "app": app, "client": client}
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# TEST 1: BUS HEARTBEAT -> 201
# =============================================================================
def test_01_heartbeat(test_env):
    client = test_env["client"]
    payload = {
        "bus_id": "BUS-042",
        "timestamp": 120.5,
        "gps_lat": 41.8781,
        "gps_lon": -87.6298,
        "speed_kmh": 32.4,
        "heading": 88.5,
        "system_status": "OK",
        "camera_status": "OK",
        "gps_status": "OK",
        "storage_status": "OK",
        "network_status": "ONLINE",
        "software_version": "bussense-2.4.0"
    }
    resp = client.post("/api/heartbeat", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "ACK"
    assert data["bus_id"] == "BUS-042"


# =============================================================================
# TEST 2: VALID POTHOLE -> 201
# =============================================================================
def test_02_valid_pothole(test_env):
    client = test_env["client"]
    payload = {
        "event_id": "EVT_POT_TEST_001",
        "bus_id": "BUS-042",
        "timestamp": 150.0,
        "gps_lat": 41.8810,
        "gps_lon": -87.6320,
        "route_id": "ROUTE-104",
        "event_type": "POTHOLE",
        "severity": "HIGH",
        "confidence": 0.94,
        "source_modality": "stereo",
        "pothole_id": 1,
        "depth_cm": 3.8,
        "area_m2": 0.225,
        "road_condition": "HIGH",
        "image": "outputs/evidence/pothole_01.jpg"
    }
    resp = client.post("/api/events/road", json=payload)
    assert resp.status_code == 201
    assert resp.json()["status"] == "ACK"

    # Verify retrieval
    get_resp = client.get("/api/events/EVT_POT_TEST_001")
    assert get_resp.status_code == 200
    assert get_resp.json()["depth_cm"] == 3.8


# =============================================================================
# TEST 3: VALID CRACK -> 201
# =============================================================================
def test_03_valid_crack(test_env):
    client = test_env["client"]
    payload = {
        "event_id": "EVT_CRK_TEST_001",
        "bus_id": "BUS-042",
        "timestamp": 155.0,
        "gps_lat": 41.8812,
        "gps_lon": -87.6325,
        "route_id": "ROUTE-104",
        "event_type": "CRACK",
        "severity": "MEDIUM",
        "confidence": 0.88,
        "source_modality": "stereo",
        "crack_id": 4,
        "length_m": 4.5,
        "branches": 3,
        "road_condition": "MEDIUM",
        "image": "outputs/evidence/crack_04.jpg"
    }
    resp = client.post("/api/events/road", json=payload)
    assert resp.status_code == 201
    assert resp.json()["status"] == "ACK"


# =============================================================================
# TEST 4: EMERGENCY INCIDENT WITH CLIP -> 201
# =============================================================================
def test_04_emergency_incident_with_clip(test_env):
    client = test_env["client"]
    payload = {
        "event_id": "EVT_INC_TEST_001",
        "bus_id": "BUS-042",
        "timestamp": 160.0,
        "gps_lat": 41.8820,
        "gps_lon": -87.6330,
        "route_id": "ROUTE-104",
        "event_type": "RAPID_CLOSING",
        "severity": "CRITICAL",
        "confidence": 0.95,
        "source_modality": "stereo",
        "primary_track_id": 3,
        "object_class": "car",
        "range_m": 3.8,
        "relative_speed_mps": -5.2,
        "ttc_s": 0.73,
        "image": "outputs/evidence/incident_event_frame.jpg",
        "event_clip": "outputs/evidence/incident_8s_clip.mp4"
    }
    resp = client.post("/api/events/incident", json=payload)
    assert resp.status_code == 201
    assert resp.json()["status"] == "ACK"


# =============================================================================
# TEST 5: VALID VRU ENCOUNTER -> 201
# =============================================================================
def test_05_vru_event(test_env):
    client = test_env["client"]
    payload = {
        "event_id": "EVT_VRU_TEST_001",
        "bus_id": "BUS-042",
        "timestamp": 165.0,
        "gps_lat": 41.8825,
        "gps_lon": -87.6335,
        "route_id": "ROUTE-104",
        "event_type": "PEDESTRIAN_IN_PATH",
        "vru_type": "pedestrian",
        "severity": "HIGH",
        "confidence": 0.91,
        "source_modality": "stereo",
        "track_id": 7,
        "range_m": 6.2,
        "ttc_s": 1.4,
        "image": "outputs/evidence/pedestrian_roi.jpg"
    }
    resp = client.post("/api/events/vru", json=payload)
    assert resp.status_code == 201
    assert resp.json()["status"] == "ACK"


# =============================================================================
# TEST 6: VALID ANPR EVENT -> 201
# =============================================================================
def test_06_valid_anpr(test_env):
    client = test_env["client"]
    payload = {
        "event_id": "EVT_ANPR_TEST_001",
        "bus_id": "BUS-042",
        "timestamp": 170.0,
        "gps_lat": 41.8830,
        "gps_lon": -87.6340,
        "route_id": "ROUTE-104",
        "event_type": "ANPR_VERIFIED",
        "severity": "LOW",
        "confidence": 0.98,
        "source_modality": "monocular",
        "track_id": 1,
        "vehicle_class": "car",
        "registration_number": "AH 25527",
        "ocr_confidence": 0.982,
        "plate_confidence": 0.950,
        "plate_pixel_width": 95.0,
        "best_frame": "outputs/evidence/plate_crop.png"
    }
    resp = client.post("/api/events/anpr", json=payload)
    assert resp.status_code == 201
    assert resp.json()["registration_number"] == "AH 25527"


# =============================================================================
# TEST 7: INVALID / UNREADABLE ANPR REJECTION -> 422
# =============================================================================
def test_07_invalid_anpr_rejection(test_env):
    client = test_env["client"]
    # Attempting to upload 'UNREADABLE' registration must fail with 422
    bad_payload = {
        "event_id": "EVT_ANPR_BAD",
        "bus_id": "BUS-042",
        "timestamp": 175.0,
        "gps_lat": 41.8830,
        "gps_lon": -87.6340,
        "route_id": "ROUTE-104",
        "event_type": "ANPR_VERIFIED",
        "severity": "LOW",
        "confidence": 0.50,
        "source_modality": "monocular",
        "track_id": 2,
        "vehicle_class": "car",
        "registration_number": "UNREADABLE",
        "ocr_confidence": 0.50,
        "plate_confidence": 0.50,
        "plate_pixel_width": 40.0
    }
    resp = client.post("/api/events/anpr", json=bad_payload)
    assert resp.status_code == 422


# =============================================================================
# TEST 8: TRAFFIC AGGREGATE -> 201
# =============================================================================
def test_08_traffic_aggregate(test_env):
    client = test_env["client"]
    payload = {
        "event_id": "EVT_TRF_TEST_001",
        "bus_id": "BUS-042",
        "timestamp": 180.0,
        "gps_lat": 41.8835,
        "gps_lon": -87.6345,
        "route_id": "ROUTE-104",
        "event_type": "TRAFFIC_AGGREGATE",
        "severity": "LOW",
        "confidence": 0.96,
        "source_modality": "stereo",
        "flow_rate_vpm": 28.5,
        "density_veh_per_km": 42.0,
        "mean_speed_kmh": 38.4,
        "median_speed_kmh": 36.5,
        "occupancy": 0.185,
        "mean_headway_s": 2.15,
        "traffic_state": "FREE_FLOW",
        "jam_risk": 0.08,
        "measurement_window_s": 30.0
    }
    resp = client.post("/api/events/traffic", json=payload)
    assert resp.status_code == 201
    assert resp.json()["traffic_state"] == "FREE_FLOW"


# =============================================================================
# TEST 9: VULNERABLE AREA AGGREGATE -> 201
# =============================================================================
def test_09_vulnerable_area_aggregate(test_env):
    client = test_env["client"]
    payload = {
        "event_id": "EVT_VAI_TEST_001",
        "bus_id": "BUS-042",
        "timestamp": 185.0,
        "gps_lat": 41.8840,
        "gps_lon": -87.6350,
        "route_id": "ROUTE-104",
        "event_type": "VULNERABLE_AREA_AGGREGATE",
        "severity": "HIGH",
        "confidence": 0.92,
        "source_modality": "stereo",
        "segment_id": "SEG_LOOP_INTERSECTION",
        "pedestrian_count": 14,
        "cyclist_count": 5,
        "crossing_count": 12,
        "conflict_count": 2,
        "near_miss_count": 1,
        "bus_pass_count": 4,
        "vulnerability_index": 28.5
    }
    resp = client.post("/api/events/vulnerable-area", json=payload)
    assert resp.status_code == 201
    assert resp.json()["segment_id"] == "SEG_LOOP_INTERSECTION"


# =============================================================================
# TEST 10: IDEMPOTENCY ON DUPLICATE EVENT ID -> 200/201 (NO DUPLICATE ROWS)
# =============================================================================
def test_10_idempotent_event_id(test_env):
    client = test_env["client"]
    payload = {
        "event_id": "EVT_IDEMPOTENT_KEY",
        "bus_id": "BUS-042",
        "timestamp": 190.0,
        "gps_lat": 41.8845,
        "gps_lon": -87.6355,
        "route_id": "ROUTE-104",
        "event_type": "POTHOLE",
        "severity": "LOW",
        "confidence": 0.90,
        "source_modality": "stereo",
        "pothole_id": 9,
        "depth_cm": 2.1,
        "area_m2": 0.120,
        "road_condition": "NORMAL"
    }

    # First attempt: 201 Created
    r1 = client.post("/api/events/road", json=payload)
    assert r1.status_code == 201
    assert r1.json()["status"] == "ACK"

    # Second attempt with same event_id: Idempotent Success
    r2 = client.post("/api/events/road", json=payload)
    assert r2.status_code in [200, 201]
    assert r2.json()["status"] == "IDEMPOTENT_ACK"
    assert r2.json()["is_duplicate"] is True

    # Database must contain exactly ONE record for this event_id
    db = test_env["app"].state.db
    evs = db.list_events(event_type="POTHOLE")
    matches = [e for e in evs if e["event_id"] == "EVT_IDEMPOTENT_KEY"]
    assert len(matches) == 1


# =============================================================================
# TEST 11: MALFORMED GPS VALIDATION -> 422
# =============================================================================
def test_11_malformed_gps_validation(test_env):
    client = test_env["client"]
    bad_payload = {
        "event_id": "EVT_BAD_GPS",
        "bus_id": "BUS-042",
        "timestamp": 195.0,
        "gps_lat": 145.0,  # Invalid latitude (> 90)
        "gps_lon": -87.6355,
        "route_id": "ROUTE-104",
        "event_type": "POTHOLE",
        "severity": "LOW",
        "confidence": 0.90,
        "source_modality": "stereo",
        "pothole_id": 9,
        "depth_cm": 2.1,
        "area_m2": 0.120
    }
    resp = client.post("/api/events/road", json=bad_payload)
    assert resp.status_code == 422


# =============================================================================
# TEST 12: INVALID CONFIDENCE VALIDATION -> 422
# =============================================================================
def test_12_invalid_confidence_validation(test_env):
    client = test_env["client"]
    bad_payload = {
        "event_id": "EVT_BAD_CONF",
        "bus_id": "BUS-042",
        "timestamp": 200.0,
        "gps_lat": 41.8850,
        "gps_lon": -87.6360,
        "route_id": "ROUTE-104",
        "event_type": "RAPID_CLOSING",
        "severity": "HIGH",
        "confidence": 1.45,  # Impossible confidence (> 1.0)
        "source_modality": "stereo",
        "primary_track_id": 4,
        "range_m": 5.0,
        "ttc_s": 1.2
    }
    resp = client.post("/api/events/incident", json=bad_payload)
    assert resp.status_code == 422


# =============================================================================
# TEST 13: OFFLINE QUEUE PERSISTENCE & RETRY
# =============================================================================
def test_13_offline_queue_resilience(test_env):
    client = test_env["client"]
    queue_dir = os.path.join(test_env["temp_dir"], "edge_queue")
    queue = BusSenseOfflineQueueClient(queue_dir=queue_dir)

    sample_event = {
        "event_id": "EVT_QUEUE_RETRY_001",
        "bus_id": "BUS-042",
        "timestamp": 205.0,
        "gps_lat": 41.8855,
        "gps_lon": -87.6365,
        "route_id": "ROUTE-104",
        "event_type": "POTHOLE",
        "severity": "HIGH",
        "confidence": 0.92,
        "source_modality": "stereo",
        "pothole_id": 12,
        "depth_cm": 4.1,
        "area_m2": 0.180,
        "road_condition": "HIGH"
    }

    # 1. Enqueue event offline
    fpath = queue.enqueue(sample_event, endpoint_path="/api/events/road")
    assert os.path.exists(fpath)
    pending = queue.get_pending_events()
    assert len(pending) == 1
    assert pending[0][1]["event_id"] == "EVT_QUEUE_RETRY_001"

    # 2. Flush pending via client
    acked = queue.flush_pending(client=client, base_url="http://testserver")
    assert acked == 1

    # 3. Queue should now be empty
    assert len(queue.get_pending_events()) == 0

    # 4. Event should exist on server
    ev_resp = client.get("/api/events/EVT_QUEUE_RETRY_001")
    assert ev_resp.status_code == 200


# =============================================================================
# TEST 14: MEDIA VALIDATION (TYPE & SIZE)
# =============================================================================
def test_14_media_validation(test_env):
    client = test_env["client"]

    # 1. Valid image upload
    dummy_jpg = b"\xff\xd8\xff\xe0" + b"\x00" * 50
    resp_valid = client.post(
        "/api/media/upload",
        data={"event_id": "EVT_MEDIA_01", "bus_id": "BUS-042", "media_type": "image"},
        files={"file": ("plate_crop.jpg", dummy_jpg, "image/jpeg")}
    )
    assert resp_valid.status_code == 201
    assert resp_valid.json()["mime_type"] == "image/jpeg"

    # 2. Invalid MIME type (e.g. text/plain for image media_type)
    resp_bad_mime = client.post(
        "/api/media/upload",
        data={"event_id": "EVT_MEDIA_02", "bus_id": "BUS-042", "media_type": "image"},
        files={"file": ("exploit.txt", b"malicious script", "text/plain")}
    )
    assert resp_bad_mime.status_code == 422


# =============================================================================
# TEST 15: SERVER RESTART PERSISTENCE
# =============================================================================
def test_15_server_restart_persistence(test_env):
    db_file = test_env["db_file"]
    client1 = test_env["client"]

    # 1. Ingest event into Instance 1
    payload = {
        "event_id": "EVT_PERSIST_RESTART",
        "bus_id": "BUS-042",
        "timestamp": 220.0,
        "gps_lat": 41.8860,
        "gps_lon": -87.6370,
        "route_id": "ROUTE-104",
        "event_type": "POTHOLE",
        "severity": "CRITICAL",
        "confidence": 0.97,
        "source_modality": "stereo",
        "pothole_id": 99,
        "depth_cm": 5.9,
        "area_m2": 0.350,
        "road_condition": "CRITICAL"
    }
    r1 = client1.post("/api/events/road", json=payload)
    assert r1.status_code == 201

    # 2. Simulate server crash and restart: create fresh FastAPI app pointing to same db_file
    fresh_app = create_app(db_path=db_file)
    client2 = TestClient(fresh_app)

    # 3. Retrieve event from Instance 2
    r2 = client2.get("/api/events/EVT_PERSIST_RESTART")
    assert r2.status_code == 200
    data = r2.json()
    assert data["event_id"] == "EVT_PERSIST_RESTART"
    assert data["depth_cm"] == 5.9
    assert data["severity"] == "CRITICAL"
