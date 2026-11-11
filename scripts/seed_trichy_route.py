"""
BusSense — Single-Bus Tiruchirappalli Route Seeder
===================================================
Wipes all existing demo data and populates the backend with
ONE bus making ONE continuous journey along the Trichy Ring Road /
Anna Salai corridor (Tiruchirappalli, Tamil Nadu, India).

Route: Thillai Nagar → Cantonment → Rockfort → Chathiram → Ariyamangalam
Real OSM road segment, ~12 km, realistic GPS progression.

Event ordering follows actual journey: potholes early, traffic building,
then incidents, VRU near-misses, then ANPR captures near Rockfort.
"""

import requests
import time
import sqlite3
import os
import uuid

BASE = "http://localhost:8000/api"
BUS_ID = "TN-BS-01"  # Tamil Nadu BusSense Unit 01
ROUTE_ID = "TRICHY-RING-001"
SOFTWARE = "bussense-2.4.0"

# ---------------------------------------------------------------------------
# Tiruchirappalli Route — real GPS waypoints along Anna Salai / Ring Road
# Ordered west→east: Thillai Nagar → Cantonment → Rockfort → Ariyamangalam
# ---------------------------------------------------------------------------
ROUTE_WAYPOINTS = [
    (10.8050, 78.6803, "Thillai Nagar Signal"),
    (10.8065, 78.6841, "LIC Colony Junction"),
    (10.8078, 78.6879, "Bharathidasan University Gate"),
    (10.8090, 78.6912, "Sri Rangam Bus Stop"),
    (10.8102, 78.6948, "Cantonment Junction"),
    (10.8115, 78.6983, "Williams Road"),
    (10.8128, 78.7018, "Rockfort Approach"),
    (10.8135, 78.7052, "Rockfort Temple Entrance"),
    (10.8142, 78.7089, "Rockfort Market"),
    (10.8138, 78.7124, "Chatram Bus Stand"),
    (10.8130, 78.7161, "Ariyamangalam Junction"),
    (10.8122, 78.7195, "Ariyamangalam Road End"),
]

def uid(prefix="EVT"):
    return f"{prefix}-{uuid.uuid4().hex[:12].upper()}"

def post(path, data):
    try:
        r = requests.post(f"{BASE}{path}", json=data, timeout=8)
        status = "OK" if r.status_code in (200, 201) else f"ERR {r.status_code}: {r.text[:80]}"
        print(f"  [{status}] {path} -> {data.get('event_id', data.get('bus_id', ''))}")
        return r
    except Exception as e:
        print(f"  [FAIL] {path} -> {e}")
        return None

def wp(index):
    """Get waypoint by index (clamped)."""
    idx = min(index, len(ROUTE_WAYPOINTS) - 1)
    return ROUTE_WAYPOINTS[idx]

def ts(minutes_ago):
    """Timestamp N minutes ago from now."""
    return time.time() - (minutes_ago * 60)

# ---------------------------------------------------------------------------
# STEP 0: Wipe existing data
# ---------------------------------------------------------------------------
def wipe_database():
    print("\n=== WIPING OLD DATA ===")
    db_path = "outputs/server_data/bussense_events.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM bus_heartbeats")
        conn.execute("DELETE FROM media_records")
        conn.commit()
        conn.close()
        print(f"  [OK] Cleared {db_path}")
    else:
        print(f"  [SKIP] DB not found at {db_path}")

# ---------------------------------------------------------------------------
# STEP 1: Bus heartbeat — current position near Rockfort (journey midpoint)
# ---------------------------------------------------------------------------
def seed_heartbeat():
    print("\n=== HEARTBEAT ===")
    lat, lon, loc = wp(7)  # Near Rockfort Temple
    post("/heartbeat", {
        "bus_id": BUS_ID,
        "timestamp": ts(1),
        "gps_lat": lat,
        "gps_lon": lon,
        "speed_kmh": 22.5,
        "heading": 92.0,
        "system_status": "OK",
        "camera_status": "OK",
        "gps_status": "OK",
        "storage_status": "OK",
        "network_status": "ONLINE",
        "software_version": SOFTWARE,
    })

# ---------------------------------------------------------------------------
# STEP 2: Potholes — early on route (Thillai Nagar area)
# ---------------------------------------------------------------------------
def seed_potholes():
    print("\n=== POTHOLES ===")
    potholes = [
        (0, "HIGH",   7.3, 0.42, "MEDIUM", 18),
        (1, "MEDIUM", 4.1, 0.28, "MEDIUM", 16),
        (2, "CRITICAL",10.5,0.65, "HIGH",  14),
        (3, "LOW",    2.2, 0.12, "NORMAL", 12),
        (4, "HIGH",   6.8, 0.38, "HIGH",   10),
    ]
    for i, (wp_idx, sev, depth, area, cond, mins_ago) in enumerate(potholes):
        lat, lon, _ = wp(wp_idx)
        post("/events/road", {
            "event_id": uid("POT"),
            "bus_id": BUS_ID,
            "timestamp": ts(mins_ago),
            "gps_lat": lat + (i * 0.0001),
            "gps_lon": lon + (i * 0.0002),
            "route_id": ROUTE_ID,
            "event_type": "POTHOLE",
            "severity": sev,
            "confidence": round(0.82 + i * 0.02, 3),
            "source_modality": "stereo",
            "software_version": SOFTWARE,
            "pothole_id": i + 1,
            "depth_cm": depth,
            "area_m2": area,
            "road_condition": cond,
        })

# ---------------------------------------------------------------------------
# STEP 3: Cracks — Cantonment area
# ---------------------------------------------------------------------------
def seed_cracks():
    print("\n=== CRACKS ===")
    cracks = [
        (3, "MEDIUM", 1.8, 2, "MEDIUM", 13),
        (4, "HIGH",   3.2, 4, "HIGH",   11),
        (5, "LOW",    0.7, 1, "NORMAL",  9),
        (5, "MEDIUM", 2.1, 3, "MEDIUM",  8),
    ]
    for i, (wp_idx, sev, length, branches, cond, mins_ago) in enumerate(cracks):
        lat, lon, _ = wp(wp_idx)
        post("/events/road", {
            "event_id": uid("CRK"),
            "bus_id": BUS_ID,
            "timestamp": ts(mins_ago),
            "gps_lat": lat + (i * 0.0001),
            "gps_lon": lon + (i * 0.0001),
            "route_id": ROUTE_ID,
            "event_type": "CRACK",
            "severity": sev,
            "confidence": round(0.74 + i * 0.03, 3),
            "source_modality": "stereo",
            "software_version": SOFTWARE,
            "crack_id": i + 1,
            "length_m": length,
            "branches": branches,
            "road_condition": cond,
        })

# ---------------------------------------------------------------------------
# STEP 4: Traffic aggregates — distributed along the route
# ---------------------------------------------------------------------------
def seed_traffic():
    print("\n=== TRAFFIC ===")
    # Progression: free flow early, building near Cantonment, congested near Rockfort
    traffic_events = [
        (1, "FREE_FLOW",  28.5, 15.2, 45.0, 43.0, 0.12, 0.04, 5.8,  0.0,  "LOW",    17),
        (2, "FREE_FLOW",  24.3, 18.7, 42.0, 40.5, 0.14, 0.05, 6.1,  0.0,  "LOW",    15),
        (3, "BUILDING",   20.1, 28.4, 32.0, 30.0, 0.28, 0.18, 4.2,  35.0, "MEDIUM", 13),
        (4, "BUILDING",   16.8, 35.1, 28.5, 26.0, 0.35, 0.31, 3.8,  60.0, "MEDIUM", 11),
        (5, "CONGESTED",  10.2, 55.3, 18.0, 16.5, 0.55, 0.58, 2.4, 140.0, "HIGH",    9),
        (6, "CONGESTED",   8.5, 68.9, 14.0, 12.0, 0.68, 0.65, 2.1, 180.0, "HIGH",    7),
        (7, "JAM_RISK",    4.1, 92.1,  8.5,  7.0, 0.82, 0.88, 1.4, 250.0, "CRITICAL", 5),
        (8, "CONGESTED",   9.8, 61.2, 16.0, 14.5, 0.60, 0.63, 2.2, 160.0, "HIGH",    3),
        (9, "BUILDING",   18.4, 32.0, 30.0, 28.0, 0.32, 0.28, 4.0,  55.0, "MEDIUM",  1),
    ]
    for i, (wp_idx, state, flow, density, mean_sp, med_sp, occ, jam, headway, queue, sev, mins_ago) in enumerate(traffic_events):
        lat, lon, _ = wp(wp_idx)
        post("/events/traffic", {
            "event_id": uid("TRF"),
            "bus_id": BUS_ID,
            "timestamp": ts(mins_ago),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": ROUTE_ID,
            "event_type": "TRAFFIC_AGGREGATE",
            "severity": sev,
            "confidence": 0.91,
            "source_modality": "stereo",
            "software_version": SOFTWARE,
            "flow_rate_vpm": flow,
            "density_veh_per_km": density,
            "mean_speed_kmh": mean_sp,
            "median_speed_kmh": med_sp,
            "occupancy": occ,
            "mean_headway_s": headway,
            "queue_length_m": queue,
            "traffic_state": state,
            "jam_risk": jam,
            "measurement_window_s": 30.0,
        })

# ---------------------------------------------------------------------------
# STEP 5: Incidents — mid-route, near Rockfort approach
# ---------------------------------------------------------------------------
def seed_incidents():
    print("\n=== INCIDENTS ===")
    incidents = [
        (5, "RAPID_CLOSING",             "HIGH",     1, "Car",   18.5, -4.2,  2.1, 10),
        (6, "DANGEROUS_FOLLOWING",        "MEDIUM",   3, "Truck", 12.0, -2.8,  2.4,  9),
        (6, "HARD_BRAKING",               "HIGH",     2, "Car",    8.5, -6.1,  1.8,  8),
        (7, "COLLISION_CANDIDATE",        "CRITICAL",  1, "Car",    4.2, -8.9,  0.7,  7),
        (7, "CUT_IN_CANDIDATE",           "HIGH",     5, "SUV",    6.0, -5.5,  1.2,  6),
        (8, "EMERGENCY_EVASIVE_MANEUVER", "CRITICAL",  2, "Truck",  3.8, -9.2,  0.5,  5),
        (9, "DANGEROUS_FOLLOWING",        "MEDIUM",   4, "Car",   15.0, -3.1,  2.2,  4),
    ]
    for i, (wp_idx, evt_type, sev, track_id, obj_cls, rng, rel_spd, ttc, mins_ago) in enumerate(incidents):
        lat, lon, _ = wp(wp_idx)
        post("/events/incident", {
            "event_id": uid("INC"),
            "bus_id": BUS_ID,
            "timestamp": ts(mins_ago),
            "gps_lat": lat + (i * 0.00005),
            "gps_lon": lon + (i * 0.00008),
            "route_id": ROUTE_ID,
            "event_type": evt_type,
            "severity": sev,
            "confidence": round(0.78 + i * 0.02, 3),
            "source_modality": "stereo",
            "software_version": SOFTWARE,
            "primary_track_id": track_id,
            "object_class": obj_cls,
            "range_m": rng,
            "relative_speed_mps": rel_spd,
            "ttc_s": ttc,
            "metric_depth_valid": True,
        })

# ---------------------------------------------------------------------------
# STEP 6: Hit-and-Run — near Rockfort Market (monocular segment)
# ---------------------------------------------------------------------------
def seed_hit_and_run():
    print("\n=== HIT-AND-RUN ===")
    lat, lon, _ = wp(8)
    post("/events/incident", {
        "event_id": uid("HAR"),
        "bus_id": BUS_ID,
        "timestamp": ts(4),
        "gps_lat": lat,
        "gps_lon": lon,
        "route_id": ROUTE_ID,
        "event_type": "HIT_AND_RUN_CANDIDATE",
        "severity": "CRITICAL",
        "confidence": 0.82,
        "source_modality": "monocular",
        "software_version": SOFTWARE,
        "primary_track_id": 1,
        "object_class": "Car",
        "metric_depth_valid": False,
        "suspect_track_id": 1,
        "victim_track_id": 4,
        "departure_detected": True,
        "registration_number": "TN 57 AH 2527",
        "ocr_confidence": 0.78,
    })

# ---------------------------------------------------------------------------
# STEP 7: VRU — near Chatram Bus Stand (pedestrian-heavy area)
# ---------------------------------------------------------------------------
def seed_vru():
    print("\n=== VRU EVENTS ===")
    vru_events = [
        (8, "PEDESTRIAN_IN_PATH",  "pedestrian", "HIGH",     3.8, 1.5, 9),
        (9, "CYCLIST_IN_PATH",     "cyclist",    "HIGH",     5.2, 2.1, 7),
        (9, "VRU_NEAR_MISS",       "pedestrian", "CRITICAL", 2.1, 0.8, 5),
        (10,"SUDDEN_VRU_EMERGENCE","pedestrian", "HIGH",     4.0, 1.8, 3),
        (10,"VRU_NEAR_MISS",       "cyclist",    "HIGH",     3.3, 1.3, 2),
    ]
    for i, (wp_idx, evt_type, vru_type, sev, rng, ttc, mins_ago) in enumerate(vru_events):
        lat, lon, _ = wp(wp_idx)
        post("/events/vru", {
            "event_id": uid("VRU"),
            "bus_id": BUS_ID,
            "timestamp": ts(mins_ago),
            "gps_lat": lat + (i * 0.00004),
            "gps_lon": lon + (i * 0.00007),
            "route_id": ROUTE_ID,
            "event_type": evt_type,
            "vru_type": vru_type,
            "severity": sev,
            "confidence": round(0.80 + i * 0.02, 3),
            "source_modality": "stereo",
            "software_version": SOFTWARE,
            "track_id": i + 10,
            "range_m": rng,
            "ttc_s": ttc,
        })

# ---------------------------------------------------------------------------
# STEP 8: ANPR — verified plates along the journey
# ---------------------------------------------------------------------------
def seed_anpr():
    print("\n=== ANPR ===")
    anpr_events = [
        (4, "TN 45 BC 1234", "Car",   0.88, 15),
        (5, "TN 57 AH 2527", "Car",   0.78,  8),  # Suspect vehicle
        (6, "TN 02 CD 5678", "Truck", 0.91, 12),
        (7, "KA 01 EF 9012", "SUV",   0.85, 10),
        (8, "TN 45 GH 3456", "Car",   0.93,  6),
        (9, "TN 63 IJ 7890", "Van",   0.87,  4),
    ]
    for i, (wp_idx, plate, veh_cls, ocr_conf, mins_ago) in enumerate(anpr_events):
        lat, lon, _ = wp(wp_idx)
        post("/events/anpr", {
            "event_id": uid("ANP"),
            "bus_id": BUS_ID,
            "timestamp": ts(mins_ago),
            "gps_lat": lat + (i * 0.00003),
            "gps_lon": lon + (i * 0.00005),
            "route_id": ROUTE_ID,
            "event_type": "ANPR_VERIFIED",
            "severity": "LOW",
            "confidence": ocr_conf,
            "source_modality": "monocular",
            "software_version": SOFTWARE,
            "track_id": i + 20,
            "vehicle_class": veh_cls,
            "registration_number": plate,
            "ocr_confidence": ocr_conf,
            "plate_confidence": round(ocr_conf + 0.03, 3),
            "plate_pixel_width": round(85.0 + i * 12, 1),
            "best_frame": f"frame_{1200 + i * 120}.jpg",
            "ocr_engine": "EasyOCR-Ensemble",
            "ocr_consensus_count": 3 + (i % 2),
        })

# ---------------------------------------------------------------------------
# STEP 9: Vulnerable areas — high-pedestrian segments
# ---------------------------------------------------------------------------
def seed_vulnerable_areas():
    print("\n=== VULNERABLE AREAS ===")
    segments = [
        (8, "SEG-ROCKFORT-MARKET", 45, 8, 18, 4, 2, 12, 68.5, "HIGH",     6),
        (9, "SEG-CHATRAM-STAND",   62, 3, 24, 5, 3,  8, 77.2, "CRITICAL",  4),
        (10,"SEG-ARIYAMANGALAM",   28, 6, 12, 2, 1,  6, 42.8, "MEDIUM",   10),
    ]
    for wp_idx, seg_id, ped, cyc, cross, conf, nm, bus_pass, vuln_idx, sev, mins_ago in segments:
        lat, lon, _ = wp(wp_idx)
        post("/events/vulnerable-area", {
            "event_id": uid("VUL"),
            "bus_id": BUS_ID,
            "timestamp": ts(mins_ago),
            "gps_lat": lat,
            "gps_lon": lon,
            "route_id": ROUTE_ID,
            "event_type": "VULNERABLE_AREA_AGGREGATE",
            "severity": sev,
            "confidence": 0.88,
            "source_modality": "stereo",
            "software_version": SOFTWARE,
            "segment_id": seg_id,
            "pedestrian_count": ped,
            "cyclist_count": cyc,
            "crossing_count": cross,
            "conflict_count": conf,
            "near_miss_count": nm,
            "bus_pass_count": bus_pass,
            "vulnerability_index": vuln_idx,
        })

if __name__ == "__main__":
    print("=" * 60)
    print("BUSSENSE — TRICHY SINGLE-BUS ROUTE SEEDER")
    print(f"Bus ID: {BUS_ID}")
    print(f"Route: {ROUTE_ID}")
    print("=" * 60)

    wipe_database()
    time.sleep(1)

    seed_heartbeat()
    seed_potholes()
    seed_cracks()
    seed_traffic()
    seed_incidents()
    seed_hit_and_run()
    seed_vru()
    seed_anpr()
    seed_vulnerable_areas()

    print("\n" + "=" * 60)
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        health = r.json()
        print(f"TOTAL EVENTS: {health['storage']['total_events']}")
        print(f"BUS ID: {BUS_ID}")
        print(f"BREAKDOWN: {health['storage']['event_breakdown']}")
    except Exception as e:
        print(f"Could not verify: {e}")
    print("=" * 60)
    print("SEEDING COMPLETE — One bus, one Trichy route.")
