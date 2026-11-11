"""
BusSense End-to-End Edge Perception to FastAPI Server Ingestion Demo
===================================================================
Demonstrates the full round-trip:
LOCAL PERCEPTION
    ↓
POTHOLE / CRACK / INCIDENT / VRU / ANPR / TRAFFIC EVENTS
    ↓
EDGE EVENT FILTER (Suppresses noise, keeps raw telemetry local)
    ↓
JSON + MEDIA PACKAGING (Representative images + ~8s clips)
    ↓
FASTAPI INGESTION
    ↓
SQLITE PERSISTENCE
    ↓
RETRIEVAL & GIS SUMMARY ENDPOINTS
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bussense_server.app import create_app
from bussense_edge.event_filter import BusSenseEdgeEventFilter
from bussense_edge.offline_queue import BusSenseOfflineQueueClient


def run_e2e_demo():
    print("=" * 70)
    print("BUSSENSE EDGE-TO-SERVER FULL INTEGRATION DEMONSTRATION")
    print("=" * 70)

    db_path = ROOT / "outputs" / "server_data" / "bussense_events.db"
    app = create_app(db_path=str(db_path))
    client = TestClient(app)

    edge_filter = BusSenseEdgeEventFilter(bus_id="BUS-042", route_id="ROUTE-104")
    queue_client = BusSenseOfflineQueueClient()

    print("\n--- PHASE 1: PERIODIC HEARTBEAT ---")
    hb = edge_filter.evaluate_heartbeat(
        timestamp=100.0, gps_lat=41.8781, gps_lon=-87.6298, speed_kmh=34.2, force=True
    )
    resp = client.post("/api/heartbeat", json=hb)
    print(f"[HTTP {resp.status_code}] Heartbeat Ingested: {resp.json()}")

    print("\n--- PHASE 2: POTHOLE DEFECT (STEREO METROLOGY) ---")
    # Simulate valid pothole measurement from potcrack_stereo
    # 1. Invalid pothole measurement: MUST BE FILTERED OUT LOCALLY
    suppressed = edge_filter.filter_pothole_defect(
        timestamp=102.0, gps_lat=41.8785, gps_lon=-87.6300,
        pothole_id=99, depth_cm=0.0, area_m2=0.0, is_valid=False
    )
    print(f"Invalid Pothole Detection: Stays local? -> {suppressed is None} (FILTERED OUT)")

    # 2. Valid pothole measurement: Uploaded with representative image
    pot_img_path = ROOT / "outputs" / "evidence" / "demo_pothole.jpg"
    pot_img_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_pot = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_pot, "POTHOLE #01: Depth 4.2cm, Area 0.24m2", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    cv2.imwrite(str(pot_img_path), dummy_pot)

    pot_event = edge_filter.filter_pothole_defect(
        timestamp=105.0, gps_lat=41.8790, gps_lon=-87.6305,
        pothole_id=1, depth_cm=4.2, area_m2=0.240, is_valid=True,
        road_condition="HIGH", representative_image_path=str(pot_img_path)
    )
    resp = client.post("/api/events/road", json=pot_event)
    print(f"[HTTP {resp.status_code}] Valid Pothole Event Ingested: {resp.json()}")

    print("\n--- PHASE 3: CRACK NETWORK EVENT ---")
    crk_event = edge_filter.filter_crack_defect(
        timestamp=110.0, gps_lat=41.8795, gps_lon=-87.6310,
        crack_id=1, length_m=5.8, branches=4, road_condition="MEDIUM"
    )
    resp = client.post("/api/events/road", json=crk_event)
    print(f"[HTTP {resp.status_code}] Crack Event Ingested: {resp.json()}")

    print("\n--- PHASE 4: EMERGENCY INCIDENT (WITH IMAGE + SHORT CLIP) ---")
    # Generate representative incident evidence frame and clip
    inc_dir = ROOT / "outputs" / "evidence" / "emergency_incident"
    inc_dir.mkdir(parents=True, exist_ok=True)
    inc_frame = inc_dir / "incident_frame.jpg"
    inc_clip = inc_dir / "incident_clip_8s.mp4"

    # Frame
    cv2.imwrite(str(inc_frame), dummy_pot)
    # 8-second clip (80 frames at 10 fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_clip = cv2.VideoWriter(str(inc_clip), fourcc, 10.0, (640, 360))
    for i in range(80):
        f = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(f, f"EMERGENCY COLLISION AVOIDANCE | Frame {i}/80", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        out_clip.write(f)
    out_clip.release()

    inc_event = edge_filter.filter_incident_event(
        timestamp=115.0, gps_lat=41.8800, gps_lon=-87.6315,
        event_type="COLLISION_CANDIDATE", primary_track_id=3,
        range_m=2.9, relative_speed_mps=-7.2, ttc_s=0.40,
        confidence=0.97, source_modality="stereo",
        image_path=str(inc_frame), clip_path=str(inc_clip)
    )
    resp = client.post("/api/events/incident", json=inc_event)
    print(f"[HTTP {resp.status_code}] Emergency Incident Event Ingested: {resp.json()}")

    # Upload actual media to server media endpoint
    with open(inc_frame, "rb") as f_img:
        media_img_resp = client.post(
            "/api/media/upload",
            data={"event_id": inc_event["event_id"], "bus_id": "BUS-042", "media_type": "image"},
            files={"file": ("incident_frame.jpg", f_img, "image/jpeg")}
        )
    print(f"[HTTP {media_img_resp.status_code}] Event Image Stored: {media_img_resp.json()['file_path']}")

    print("\n--- PHASE 5: VRU ENCOUNTER ---")
    vru_event = edge_filter.filter_vru_event(
        timestamp=120.0, gps_lat=41.8805, gps_lon=-87.6320,
        event_type="VRU_NEAR_MISS", track_id=8, vru_type="pedestrian",
        range_m=4.8, ttc_s=0.85, confidence=0.94
    )
    resp = client.post("/api/events/vru", json=vru_event)
    print(f"[HTTP {resp.status_code}] VRU Near-Miss Ingested: {resp.json()}")

    print("\n--- PHASE 6: ANPR (REJECT UNREADABLE, UPLOAD VERIFIED) ---")
    # 1. Unreadable OCR: Stays silent and local
    unreadable = edge_filter.filter_anpr_event(
        timestamp=122.0, gps_lat=41.8810, gps_lon=-87.6325,
        track_id=2, registration_number="UNREADABLE",
        ocr_confidence=0.20, plate_confidence=0.30, plate_pixel_width=35.0
    )
    print(f"Unreadable OCR: Stays local? -> {unreadable is None} (SUPPRESSED)")

    # 2. Verified OCR: Uploaded
    anpr_event = edge_filter.filter_anpr_event(
        timestamp=125.0, gps_lat=41.8815, gps_lon=-87.6330,
        track_id=1, registration_number="AH 25527",
        ocr_confidence=0.982, plate_confidence=0.950, plate_pixel_width=95.0,
        best_frame_path="outputs/evidence/hit_and_run/EV_FELONY_HIT_AND_RUN_F4/OCR_visualization.png"
    )
    resp = client.post("/api/events/anpr", json=anpr_event)
    print(f"[HTTP {resp.status_code}] Verified ANPR Ingested: {resp.json()}")

    print("\n--- PHASE 7: TRAFFIC DYNAMICS AGGREGATE ---")
    trf_event = edge_filter.filter_traffic_aggregate(
        timestamp=130.0, gps_lat=41.8820, gps_lon=-87.6335,
        flow_rate_vpm=26.4, density_veh_per_km=42.5,
        mean_speed_kmh=36.2, median_speed_kmh=35.0,
        occupancy=0.175, traffic_state="CONGESTED", jam_risk=0.68,
        mean_headway_s=2.25, queue_length_m=45.0
    )
    resp = client.post("/api/events/traffic", json=trf_event)
    print(f"[HTTP {resp.status_code}] Traffic Aggregate Ingested: {resp.json()}")

    print("\n--- PHASE 8: SPATIAL RETRIEVAL & QUERY VERIFICATION ---")
    # Query all events
    all_evs = client.get("/api/events").json()
    print(f"Total Server Ingested Events: {len(all_evs)}")

    # Query specific emergency incident
    retrieved_inc = client.get(f"/api/events/{inc_event['event_id']}").json()
    print(f"Retrieved Emergency Incident by ID: {retrieved_inc['event_id']}")
    print(f"  - Severity:           {retrieved_inc['severity']}")
    print(f"  - TTC:                {retrieved_inc['ttc_s']} s")
    print(f"  - Range:              {retrieved_inc['range_m']} m")
    print(f"  - Image Attachment:   {retrieved_inc['image']}")
    print(f"  - Clip Attachment:    {retrieved_inc['event_clip']}")

    # Query Road Defect Layer
    road_evs = client.get("/api/road-events").json()
    print(f"Road Defect Events Layer: {len(road_evs)} defect records")

    # Query Health
    health = client.get("/api/health").json()
    print(f"Server Health & Storage State: {health['storage']}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_demo()
