"""
BusSense 24h Telemetry & Event Timeline Distributor (MySQL & SQLite Multi-Dialect)
================================================================================
Distributes perception events across the rolling 24-hour window relative to the current time,
and injects realistic periodic events so the 24h Perception Event Dynamics timeline renders
an active, continuous detection curve across all 24 hour buckets.
"""

import time
import random
import json
import sqlite3
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_server.db_config import SessionLocal, get_database_url
from bussense_server.models import EventModel, InfrastructureEvent, BusHeartbeat

SQLITE_DB_PATH = ROOT / "outputs" / "server_data" / "bussense_events.db"


def distribute_and_seed_24h_events():
    now = time.time()
    random.seed(int(now) % 100000)

    print(f"[INFO] Current Epoch Time: {now:.2f} ({datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat()})")

    # -------------------------------------------------------------------------
    # 1. Update Existing Canonical Events in MySQL
    # -------------------------------------------------------------------------
    session = SessionLocal()
    try:
        events = session.query(EventModel).order_by(EventModel.event_id).all()
        print(f"[INFO] Found {len(events)} existing events in primary database.")

        traffic_hours = [0.8, 1.5, 2.8, 4.2, 5.5, 7.0, 8.5, 10.2, 11.8, 13.5, 15.0, 16.8, 18.2, 20.0, 21.5, 22.8]
        defect_hours = [1.2, 3.4, 6.1, 8.9, 11.2, 14.1, 17.3, 19.5, 22.1]
        incident_hours = [0.5, 2.1, 3.8, 5.2, 7.5, 9.1, 10.8, 12.6, 14.9, 16.2, 17.9, 18.8, 20.6, 22.4]
        vru_hours = [1.8, 4.5, 8.1, 9.6, 13.1, 16.5, 17.5, 21.0]
        anpr_hours = [2.5, 6.8, 10.5, 14.5, 18.5, 22.0]
        infra_hours = [3.1, 5.8, 7.8, 11.5, 13.8, 15.8, 19.1, 21.2, 23.1]

        t_idx, d_idx, i_idx, v_idx, a_idx, o_idx = 0, 0, 0, 0, 0, 0

        for ev in events:
            ev_type = (ev.event_type or "").upper()
            if "TRAFFIC" in ev_type:
                h = traffic_hours[t_idx % len(traffic_hours)]
                t_idx += 1
            elif ev_type in ("POTHOLE", "CRACK"):
                h = defect_hours[d_idx % len(defect_hours)]
                d_idx += 1
            elif "INCIDENT" in ev_type or ev_type in (
                "COLLISION_CANDIDATE", "DANGEROUS_FOLLOWING", "EMERGENCY_EVASIVE_MANEUVER",
                "HARD_BRAKING", "HIT_AND_RUN_CANDIDATE", "RAPID_CLOSING", "WEAVING_CANDIDATE",
                "CUT_IN_CANDIDATE", "ABRUPT_LATERAL_MANEUVER"
            ):
                h = incident_hours[i_idx % len(incident_hours)]
                i_idx += 1
            elif "VRU" in ev_type or "PEDESTRIAN" in ev_type:
                h = vru_hours[v_idx % len(vru_hours)]
                v_idx += 1
            elif "ANPR" in ev_type:
                h = anpr_hours[a_idx % len(anpr_hours)]
                a_idx += 1
            else:
                h = infra_hours[o_idx % len(infra_hours)]
                o_idx += 1

            jitter = random.uniform(-400, 400)
            offset_sec = max(180, min(23.7 * 3600, h * 3600 + jitter))
            new_ts = now - offset_sec
            new_iso = datetime.datetime.fromtimestamp(new_ts, datetime.timezone.utc).isoformat()

            ev.timestamp = new_ts
            ev.created_at = new_iso

            try:
                pj = json.loads(ev.payload_json) if ev.payload_json else {}
                pj["timestamp"] = new_ts
                ev.payload_json = json.dumps(pj)
            except Exception:
                pass

        # Update infrastructure events
        infra_evs = session.query(InfrastructureEvent).all()
        for iev in infra_evs:
            parent_ev = session.query(EventModel).filter(EventModel.event_id == iev.event_id).first()
            if parent_ev:
                iev.last_seen = parent_ev.timestamp
                iev.first_seen = parent_ev.timestamp - 1800
                iev.updated_at = parent_ev.created_at

        # ---------------------------------------------------------------------
        # 2. Inject Additional Randomized Perception Events across 24h
        # ---------------------------------------------------------------------
        additional_types = [
            ("TRAFFIC_AGGREGATE", "LOW", "CAMERA_STEREO", "Moderate traffic flow observed"),
            ("TRAFFIC_AGGREGATE", "MEDIUM", "CAMERA_STEREO", "Dense urban corridor progression"),
            ("VRU_PEDESTRIAN", "LOW", "CAMERA_FRONT", "Pedestrian walking on corridor shoulder"),
            ("VRU_NEAR_MISS", "HIGH", "CAMERA_FRONT", "Cyclist merging near bus corridor"),
            ("POTHOLE", "MEDIUM", "CAMERA_STEREO", "Pavement cavity detected, depth 3.8 cm"),
            ("CRACK", "LOW", "CAMERA_STEREO", "Surface transverse crack network 1.4 m"),
            ("ANPR_VERIFIED", "LOW", "CAMERA_OCR", "Commercial transit plate verified"),
            ("ROAD_HAZARD", "MEDIUM", "CAMERA_FRONT", "Unmarked construction barrier near curb"),
            ("LANE_MARKING_HEALTH", "LOW", "CAMERA_FRONT", "Corridor center line wear 28%"),
            ("TRAFFIC_SIGN_OBSERVATION", "LOW", "CAMERA_FRONT", "Speed limit 40 km/h sign confirmed legible"),
            ("WATERLOGGING_SECTION", "MEDIUM", "CAMERA_FRONT", "Shoulder surface ponding length 8.2m"),
            ("HARD_BRAKING", "HIGH", "TELEMETRY_IMU", "Evasive deceleration event 0.42g"),
        ]

        # Add 30 distributed random events to ensure all 24 hour buckets have points
        for i in range(30):
            event_id = f"EVT-AUTO-24H-{i+1:03d}"
            existing = session.query(EventModel).filter(EventModel.event_id == event_id).first()
            
            ev_type, sev, mod, desc = random.choice(additional_types)
            offset_h = (i / 30.0) * 23.5 + random.uniform(0.05, 0.4)
            ev_ts = now - (offset_h * 3600)
            ev_iso = datetime.datetime.fromtimestamp(ev_ts, datetime.timezone.utc).isoformat()
            
            lat = 10.8000 + random.uniform(0.005, 0.035)
            lon = 78.6850 + random.uniform(0.005, 0.025)

            payload = {
                "event_id": event_id,
                "event_type": ev_type,
                "severity": sev,
                "timestamp": ev_ts,
                "bus_id": "BUS-01",
                "route_id": "ROUTE-TRICHY-01",
                "confidence": round(random.uniform(0.85, 0.99), 2),
                "description": desc,
                "gps": {"latitude": lat, "longitude": lon},
            }

            if existing:
                existing.timestamp = ev_ts
                existing.created_at = ev_iso
                existing.payload_json = json.dumps(payload)
                existing.confidence = payload["confidence"]
            else:
                new_event = EventModel(
                    event_id=event_id,
                    bus_id="BUS-01",
                    timestamp=ev_ts,
                    gps_lat=lat,
                    gps_lon=lon,
                    route_id="ROUTE-TRICHY-01",
                    event_type=ev_type,
                    severity=sev,
                    confidence=payload["confidence"],
                    source_modality=mod,
                    software_version="bussense-2.4.0",
                    road_segment_id="SEG-TRICHY-CORRIDOR",
                    payload_json=json.dumps(payload),
                    created_at=ev_iso,
                )
                session.add(new_event)

        session.commit()
        print("[SUCCESS] MySQL events successfully updated and seeded across 24h window!")

    except Exception as e:
        session.rollback()
        print(f"[ERROR] Error updating MySQL database: {e}")
    finally:
        session.close()

    # -------------------------------------------------------------------------
    # 3. Mirror to SQLite for Full Multi-Dialect Parity
    # -------------------------------------------------------------------------
    if SQLITE_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(SQLITE_DB_PATH))
            cur = conn.cursor()
            
            # Read all updated events from MySQL and mirror to SQLite
            session2 = SessionLocal()
            all_evs = session2.query(EventModel).all()
            for ev in all_evs:
                cur.execute(
                    """
                    INSERT INTO events (event_id, bus_id, timestamp, gps_lat, gps_lon, route_id, event_type, severity, confidence, source_modality, software_version, road_segment_id, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        timestamp = excluded.timestamp,
                        payload_json = excluded.payload_json,
                        created_at = excluded.created_at
                    """,
                    (ev.event_id, ev.bus_id, ev.timestamp, ev.gps_lat, ev.gps_lon, ev.route_id, ev.event_type, ev.severity, ev.confidence, ev.source_modality, ev.software_version, ev.road_segment_id, ev.payload_json, ev.created_at)
                )
            conn.commit()
            conn.close()
            session2.close()
            print("[SUCCESS] SQLite backup mirrored with exact 24h event timeline.")
        except Exception as e:
            print(f"[WARN] SQLite mirror skipped or encountered error: {e}")


if __name__ == "__main__":
    distribute_and_seed_24h_events()
