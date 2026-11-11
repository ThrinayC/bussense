"""
BusSense Demo Data Seeder
=========================
Seeds the backend with realistic multi-bus, multi-route events
for a convincing dashboard demonstration.
"""

import requests
import time
import random
import uuid

BASE = "http://localhost:8000/api"

# Chicago-area GPS coordinates for realistic map pins
CHICAGO_LOCATIONS = [
    (41.8781, -87.6298, "Downtown Loop"),
    (41.8827, -87.6233, "Millennium Park"),
    (41.8919, -87.6087, "Navy Pier"),
    (41.8525, -87.6514, "Chinatown"),
    (41.9032, -87.6267, "Lincoln Park"),
    (41.8675, -87.6166, "Grant Park"),
    (41.8823, -87.6404, "Union Station"),
    (41.8827, -87.6355, "Willis Tower"),
    (41.8986, -87.6229, "Old Town"),
    (41.8758, -87.6189, "Museum Campus"),
    (41.9114, -87.6345, "Wrigleyville"),
    (41.8488, -87.6175, "McCormick Place"),
    (41.8640, -87.6498, "Pilsen"),
    (41.8943, -87.6385, "Goose Island"),
    (41.8725, -87.6278, "South Loop"),
]

BUS_IDS = ["BUS-001", "BUS-007", "BUS-012", "BUS-019", "BUS-025", "BUS-033", "BUS-041"]
ROUTES = ["RT-BLUE", "RT-RED", "RT-GREEN", "RT-ORANGE", "RT-BROWN"]


def uid(prefix="EVT"):
    return f"{prefix}-{uuid.uuid4().hex[:12].upper()}"


def loc():
    lat, lon, name = random.choice(CHICAGO_LOCATIONS)
    # Add slight jitter
    return lat + random.uniform(-0.005, 0.005), lon + random.uniform(-0.005, 0.005), name


def post(path, data):
    try:
        r = requests.post(f"{BASE}{path}", json=data, timeout=5)
        status = "OK" if r.status_code in (200, 201) else f"ERR {r.status_code}"
        print(f"  [{status}] {path} -> {data.get('event_id', data.get('bus_id', ''))}")
        return r
    except Exception as e:
        print(f"  [FAIL] {path} -> {e}")
        return None


def seed_heartbeats():
    print("\n=== HEARTBEATS ===")
    for bus in BUS_IDS:
        lat, lon, _ = loc()
        post("/heartbeat", {
            "bus_id": bus,
            "timestamp": time.time() - random.randint(0, 300),
            "gps_lat": lat,
            "gps_lon": lon,
            "speed_kmh": round(random.uniform(0, 55), 1),
            "heading": round(random.uniform(0, 360), 1),
            "system_status": random.choice(["OK", "OK", "OK", "WARNING"]),
            "camera_status": "OK",
            "gps_status": "OK",
            "storage_status": random.choice(["OK", "OK", "NEAR_FULL"]),
            "network_status": "ONLINE",
            "software_version": "bussense-2.4.0"
        })


def seed_potholes():
    print("\n=== POTHOLES ===")
    severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    conditions = ["NORMAL", "MEDIUM", "HIGH", "CRITICAL"]
    for i in range(12):
        lat, lon, _ = loc()
        sev = random.choice(severities)
        post("/events/road", {
            "event_id": uid("POT"),
            "bus_id": random.choice(BUS_IDS),
            "timestamp": time.time() - random.randint(300, 86400),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": random.choice(ROUTES),
            "event_type": "POTHOLE",
            "severity": sev,
            "confidence": round(random.uniform(0.72, 0.98), 3),
            "source_modality": "stereo",
            "pothole_id": i + 1,
            "depth_cm": round(random.uniform(1.5, 12.0), 2),
            "area_m2": round(random.uniform(0.05, 1.2), 3),
            "road_condition": random.choice(conditions),
            "software_version": "bussense-2.4.0"
        })


def seed_cracks():
    print("\n=== CRACKS ===")
    for i in range(8):
        lat, lon, _ = loc()
        post("/events/road", {
            "event_id": uid("CRK"),
            "bus_id": random.choice(BUS_IDS),
            "timestamp": time.time() - random.randint(300, 86400),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": random.choice(ROUTES),
            "event_type": "CRACK",
            "severity": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "confidence": round(random.uniform(0.65, 0.95), 3),
            "source_modality": "stereo",
            "crack_id": i + 1,
            "length_m": round(random.uniform(0.3, 4.0), 2),
            "branches": random.randint(0, 5),
            "road_condition": random.choice(["NORMAL", "MEDIUM", "HIGH"]),
            "software_version": "bussense-2.4.0"
        })


def seed_incidents():
    print("\n=== INCIDENTS ===")
    types = [
        "RAPID_CLOSING", "DANGEROUS_FOLLOWING", "HARD_BRAKING",
        "CUT_IN_CANDIDATE", "COLLISION_CANDIDATE", "EMERGENCY_EVASIVE_MANEUVER"
    ]
    for i in range(10):
        lat, lon, _ = loc()
        evt = random.choice(types)
        post("/events/incident", {
            "event_id": uid("INC"),
            "bus_id": random.choice(BUS_IDS),
            "timestamp": time.time() - random.randint(60, 43200),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": random.choice(ROUTES),
            "event_type": evt,
            "severity": random.choice(["MEDIUM", "HIGH", "CRITICAL"]),
            "confidence": round(random.uniform(0.70, 0.96), 3),
            "source_modality": "stereo",
            "primary_track_id": random.randint(1, 50),
            "object_class": random.choice(["Car", "Truck", "SUV", "Van"]),
            "range_m": round(random.uniform(3.0, 25.0), 2),
            "relative_speed_mps": round(random.uniform(-8.0, -1.0), 2),
            "ttc_s": round(random.uniform(0.3, 2.4), 2),
            "metric_depth_valid": True,
            "software_version": "bussense-2.4.0"
        })


def seed_hit_and_run():
    print("\n=== HIT-AND-RUN ===")
    for i in range(3):
        lat, lon, _ = loc()
        post("/events/incident", {
            "event_id": uid("HAR"),
            "bus_id": random.choice(BUS_IDS),
            "timestamp": time.time() - random.randint(1800, 72000),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": random.choice(ROUTES),
            "event_type": "HIT_AND_RUN_CANDIDATE",
            "severity": "CRITICAL",
            "confidence": round(random.uniform(0.65, 0.88), 3),
            "source_modality": "monocular",
            "primary_track_id": random.randint(1, 30),
            "object_class": random.choice(["Car", "SUV"]),
            "metric_depth_valid": False,
            "suspect_track_id": random.randint(1, 30),
            "victim_track_id": random.randint(31, 50),
            "departure_detected": True,
            "registration_number": random.choice(["AH 25527", "IL 7834KT", "CH 90124"]),
            "ocr_confidence": round(random.uniform(0.55, 0.85), 3),
            "software_version": "bussense-2.4.0"
        })


def seed_vru():
    print("\n=== VRU EVENTS ===")
    types = ["PEDESTRIAN_IN_PATH", "CYCLIST_IN_PATH", "VRU_NEAR_MISS", "SUDDEN_VRU_EMERGENCE"]
    vru_types = ["pedestrian", "cyclist", "rider"]
    for i in range(8):
        lat, lon, _ = loc()
        post("/events/vru", {
            "event_id": uid("VRU"),
            "bus_id": random.choice(BUS_IDS),
            "timestamp": time.time() - random.randint(120, 36000),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": random.choice(ROUTES),
            "event_type": random.choice(types),
            "vru_type": random.choice(vru_types),
            "severity": random.choice(["MEDIUM", "HIGH", "CRITICAL"]),
            "confidence": round(random.uniform(0.75, 0.95), 3),
            "source_modality": "stereo",
            "track_id": random.randint(1, 40),
            "range_m": round(random.uniform(1.5, 12.0), 2),
            "ttc_s": round(random.uniform(0.5, 3.5), 2),
            "software_version": "bussense-2.4.0"
        })


def seed_anpr():
    print("\n=== ANPR ===")
    plates = [
        "AH 25527", "IL 7834KT", "CH 90124", "KA 51MG 2847",
        "MH 12DE 1433", "TN 09AB 5562", "DL 4CAF 7789",
        "GJ 01AB 3344", "RJ 14CD 8821", "UP 32EF 5567"
    ]
    for i, plate in enumerate(plates):
        lat, lon, _ = loc()
        post("/events/anpr", {
            "event_id": uid("ANP"),
            "bus_id": random.choice(BUS_IDS),
            "timestamp": time.time() - random.randint(60, 28800),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": random.choice(ROUTES),
            "event_type": "ANPR_VERIFIED",
            "severity": "LOW",
            "confidence": round(random.uniform(0.80, 0.98), 3),
            "source_modality": "monocular",
            "track_id": random.randint(1, 60),
            "vehicle_class": random.choice(["Car", "Truck", "SUV", "Van", "Bus"]),
            "registration_number": plate,
            "ocr_confidence": round(random.uniform(0.65, 0.95), 3),
            "plate_confidence": round(random.uniform(0.70, 0.95), 3),
            "plate_pixel_width": round(random.uniform(45, 180), 1),
            "best_frame": f"frame_{random.randint(100, 2400)}.jpg",
            "ocr_engine": "EasyOCR-Ensemble",
            "ocr_consensus_count": random.randint(2, 5),
            "software_version": "bussense-2.4.0"
        })


def seed_traffic():
    print("\n=== TRAFFIC ===")
    states = ["FREE_FLOW", "BUILDING", "CONGESTED", "JAM_RISK"]
    for i in range(15):
        lat, lon, _ = loc()
        state = random.choice(states)
        jam = {"FREE_FLOW": 0.05, "BUILDING": 0.35, "CONGESTED": 0.65, "JAM_RISK": 0.90}[state]
        post("/events/traffic", {
            "event_id": uid("TRF"),
            "bus_id": random.choice(BUS_IDS),
            "timestamp": time.time() - random.randint(30, 7200),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": random.choice(ROUTES),
            "event_type": "TRAFFIC_AGGREGATE",
            "severity": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "confidence": round(random.uniform(0.80, 0.98), 3),
            "source_modality": "stereo",
            "flow_rate_vpm": round(random.uniform(2, 40), 1),
            "density_veh_per_km": round(random.uniform(5, 120), 1),
            "mean_speed_kmh": round(random.uniform(5, 60), 1),
            "median_speed_kmh": round(random.uniform(5, 55), 1),
            "occupancy": round(random.uniform(0.05, 0.85), 3),
            "mean_headway_s": round(random.uniform(0.8, 6.0), 2),
            "queue_length_m": round(random.uniform(0, 250), 1),
            "traffic_state": state,
            "jam_risk": round(jam + random.uniform(-0.1, 0.1), 3),
            "measurement_window_s": 30.0,
            "software_version": "bussense-2.4.0"
        })


def seed_vulnerable_areas():
    print("\n=== VULNERABLE AREAS ===")
    segments = ["SEG-LOOP-N", "SEG-LOOP-S", "SEG-NAVY-PIER", "SEG-LINCOLN", "SEG-WRIGLEY", "SEG-PILSEN"]
    for seg in segments:
        lat, lon, _ = loc()
        post("/events/vulnerable-area", {
            "event_id": uid("VUL"),
            "bus_id": random.choice(BUS_IDS),
            "timestamp": time.time() - random.randint(600, 14400),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": random.choice(ROUTES),
            "event_type": "VULNERABLE_AREA_AGGREGATE",
            "severity": random.choice(["MEDIUM", "HIGH", "CRITICAL"]),
            "confidence": round(random.uniform(0.80, 0.95), 3),
            "source_modality": "stereo",
            "segment_id": seg,
            "pedestrian_count": random.randint(5, 80),
            "cyclist_count": random.randint(0, 25),
            "crossing_count": random.randint(2, 30),
            "conflict_count": random.randint(0, 8),
            "near_miss_count": random.randint(0, 4),
            "bus_pass_count": random.randint(3, 20),
            "vulnerability_index": round(random.uniform(10, 85), 1),
            "software_version": "bussense-2.4.0"
        })


if __name__ == "__main__":
    print("=" * 60)
    print("BUSSENSE DEMO DATA SEEDER")
    print("=" * 60)

    seed_heartbeats()
    seed_potholes()
    seed_cracks()
    seed_incidents()
    seed_hit_and_run()
    seed_vru()
    seed_anpr()
    seed_traffic()
    seed_vulnerable_areas()

    print("\n" + "=" * 60)
    # Verify
    r = requests.get(f"{BASE}/health")
    health = r.json()
    print(f"TOTAL EVENTS: {health['storage']['total_events']}")
    print(f"BREAKDOWN: {health['storage']['event_breakdown']}")
    print("=" * 60)
    print("SEEDING COMPLETE")
