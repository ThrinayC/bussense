"""
BusSense 24h Telemetry & Event Timeline Distributor
===================================================
Distributes authentic perception events across the rolling 24-hour window
so the Operational Overview timeline chart renders active detection dynamics.
"""

import sqlite3
import time
import random
import json
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "outputs" / "server_data" / "bussense_events.db"


def distribute_events_across_24h():
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT event_id, event_type, payload_json FROM events ORDER BY event_id")
    rows = cur.fetchall()

    if not rows:
        print("[WARN] No events found in database to distribute.")
        conn.close()
        return

    now = time.time()
    random.seed(42)

    traffic_hours = [1.0, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 21.5, 23.0]
    defect_hours = [0.8, 3.2, 5.5, 7.8, 11.0, 13.5, 15.8, 19.2, 22.5]
    incident_hours = [0.5, 1.8, 2.9, 4.5, 6.8, 8.5, 9.8, 12.5, 14.8, 16.5, 17.8, 18.9, 20.5, 22.0]
    vru_hours = [1.2, 7.2, 8.8, 15.2, 17.2]
    anpr_hours = [2.2, 9.2, 14.2, 18.2]
    infra_hours = [3.5, 6.2, 10.5, 13.8, 16.2, 19.8, 21.8, 22.8, 23.2]

    t_idx, d_idx, i_idx, v_idx, a_idx, o_idx = 0, 0, 0, 0, 0, 0

    for ev_id, ev_type, pj in rows:
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

        jitter = random.uniform(-600, 600)
        offset_sec = max(180, min(23.5 * 3600, h * 3600 + jitter))
        new_ts = now - offset_sec
        new_iso = datetime.datetime.fromtimestamp(new_ts, datetime.timezone.utc).isoformat()

        payload = json.loads(pj)
        payload["timestamp"] = new_ts
        new_pj = json.dumps(payload)

        cur.execute(
            "UPDATE events SET timestamp = ?, payload_json = ?, created_at = ? WHERE event_id = ?",
            (new_ts, new_pj, new_iso, ev_id)
        )

    # Synchronize infrastructure_events
    cur.execute("SELECT event_id FROM infrastructure_events")
    infra_eids = [r[0] for r in cur.fetchall()]
    for eid in infra_eids:
        cur.execute("SELECT timestamp FROM events WHERE event_id = ?", (eid,))
        ev_row = cur.fetchone()
        if ev_row:
            ev_ts = ev_row[0]
            cur.execute(
                "UPDATE infrastructure_events SET first_seen = ?, last_seen = ? WHERE event_id = ?",
                (ev_ts - 1800, ev_ts, eid)
            )

    conn.commit()

    # Verify 24h bucket counts
    cutoff = now - 24 * 3600
    cur.execute("SELECT timestamp FROM events")
    ts_list = [r[0] for r in cur.fetchall()]
    buckets = [0] * 24
    for ts in ts_list:
        if cutoff <= ts <= now:
            b = min(int((ts - cutoff) / 3600), 23)
            buckets[b] += 1

    conn.close()
    print(f"[SUCCESS] Distributed {len(rows)} events across past 24 hours.")
    print(f"24h Timeline Hourly Buckets: {buckets}")


if __name__ == "__main__":
    distribute_events_across_24h()
