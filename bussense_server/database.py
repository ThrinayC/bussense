"""
BusSense Server Database — SQLite Persistence & Idempotency Layer
==================================================================
Provides atomic persistence, duplicate event suppression (idempotency),
and deterministic media path resolution per Sections 14, 17, 18, and 20.
"""

import os
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import time


class BusSenseDatabase:
    """
    Lightweight SQLite storage engine for BusSense events, heartbeats, and media.
    Persists data to disk across server restarts (validates TEST 15).
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            root_dir = Path(__file__).resolve().parent.parent
            data_dir = root_dir / "outputs" / "server_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "bussense_events.db")
        else:
            self.db_path = db_path
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cur = conn.cursor()
            # 1. Main Events Table (Envelope + Type-specific payload)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                bus_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                gps_lat REAL,
                gps_lon REAL,
                route_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_modality TEXT NOT NULL,
                software_version TEXT NOT NULL,
                road_segment_id TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)

            # 2. Bus Heartbeats Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS bus_heartbeats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bus_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                gps_lat REAL,
                gps_lon REAL,
                speed_kmh REAL NOT NULL,
                system_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                received_at TEXT NOT NULL
            );
            """)

            # 3. Media Registry Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS media_records (
                media_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                bus_id TEXT NOT NULL,
                media_type TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                checksum_sha256 TEXT NOT NULL,
                file_path TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES events (event_id)
            );
            """)

            # 4. CSR Partners Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS csr_partners (
                company_id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                logo TEXT,
                csr_budget_inr REAL NOT NULL,
                available_budget_inr REAL NOT NULL,
                focus_areas TEXT NOT NULL,
                geography TEXT NOT NULL,
                contact_name TEXT NOT NULL,
                contact_email TEXT NOT NULL,
                contact_phone TEXT NOT NULL,
                website TEXT,
                demo_flag INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            );
            """)

            # 5. CSR Requests Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS csr_requests (
                request_id TEXT PRIMARY KEY,
                company_id TEXT NOT NULL,
                pothole_event_id TEXT NOT NULL,
                bus_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                gps_lat REAL,
                gps_lon REAL,
                depth_cm REAL,
                area_m2 REAL,
                severity TEXT,
                road_name TEXT,
                estimated_repair_cost_inr REAL NOT NULL,
                requested_amount_inr REAL NOT NULL,
                evidence_image TEXT,
                notes TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (company_id) REFERENCES csr_partners (company_id),
                FOREIGN KEY (pothole_event_id) REFERENCES events (event_id)
            );
            """)

            # 6. Reference Assets Table (Section 15)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS reference_assets (
                asset_id TEXT PRIMARY KEY,
                road_segment_id TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                asset_subtype TEXT,
                expected_state TEXT NOT NULL,
                reference_source TEXT NOT NULL,
                source_quality TEXT DEFAULT 'DEMO',
                geometry_json TEXT,
                allowed_openings_json TEXT,
                last_verified TEXT,
                created_at TEXT NOT NULL
            );
            """)

            # 7. Infrastructure Observations Table (Section 18)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS infrastructure_observations (
                observation_id TEXT PRIMARY KEY,
                bus_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                latitude REAL,
                longitude REAL,
                heading REAL,
                road_segment_id TEXT,
                event_type TEXT NOT NULL,
                track_id INTEGER,
                model_confidence REAL NOT NULL,
                visibility_score REAL NOT NULL,
                temporal_consistency REAL DEFAULT 1.0,
                parameters_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL,
                simulation_mode INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """)

            # 8. Infrastructure Events Table (Section 19)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS infrastructure_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                road_segment_id TEXT,
                reference_asset_id TEXT,
                status TEXT NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                observation_count INTEGER DEFAULT 1,
                unique_bus_count INTEGER DEFAULT 1,
                unique_pass_count INTEGER DEFAULT 1,
                unique_day_count INTEGER DEFAULT 1,
                visibility_summary_json TEXT,
                temporal_consistency REAL DEFAULT 1.0,
                spatial_consistency REAL DEFAULT 1.0,
                reference_support TEXT DEFAULT 'UNKNOWN',
                conflict_count INTEGER DEFAULT 0,
                parameters_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL,
                simulation_mode INTEGER DEFAULT 0,
                latitude REAL,
                longitude REAL,
                heading REAL,
                bus_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 9. Bus Road-Constrained Telemetry Table (Section 23.6)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS bus_telemetry (
                telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                bus_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                heading REAL NOT NULL,
                road_name TEXT NOT NULL,
                osm_way_id INTEGER,
                road_segment_id TEXT NOT NULL,
                distance_along_road_m REAL NOT NULL,
                distance_to_centerline_m REAL NOT NULL,
                speed_kmh REAL NOT NULL,
                simulation_mode INTEGER DEFAULT 1,
                recorded_at TEXT NOT NULL
            );
            """)

            # Indexes for high-speed dashboard and GIS queries
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_bus ON events(bus_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_heartbeats_bus ON bus_heartbeats(bus_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_csr_requests_pothole ON csr_requests(pothole_event_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_csr_requests_company ON csr_requests(company_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_csr_requests_status ON csr_requests(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_infra_obs_type ON infrastructure_observations(event_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_infra_obs_bus ON infrastructure_observations(bus_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_infra_events_type ON infrastructure_events(event_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_infra_events_status ON infrastructure_events(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_infra_events_seg ON infrastructure_events(road_segment_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ref_assets_type ON reference_assets(asset_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_bus_telem_bus ON bus_telemetry(bus_id);")

            self._seed_default_csr_data(cur)
            conn.commit()

    # =========================================================================
    # EVENT INGESTION & IDEMPOTENCY
    # =========================================================================

    def insert_event(self, event_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Inserts an event into the database.
        Returns: (is_new: bool, stored_record: dict)
        If event_id already exists, returns (False, existing_record) without error (Idempotency).
        """
        event_id = event_data["event_id"]
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT payload_json FROM events WHERE event_id = ?", (event_id,))
            row = cur.fetchone()
            if row:
                # Idempotent response: return existing record
                return False, json.loads(row["payload_json"])

            # Insert new record
            now_str = datetime.now(timezone.utc).isoformat()
            lat_val = float(event_data["gps_lat"]) if event_data.get("gps_lat") is not None else None
            lon_val = float(event_data["gps_lon"]) if event_data.get("gps_lon") is not None else None
            cur.execute("""
            INSERT INTO events (
                event_id, bus_id, timestamp, gps_lat, gps_lon, route_id,
                event_type, severity, confidence, source_modality, software_version,
                road_segment_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                event_data["bus_id"],
                float(event_data["timestamp"]),
                lat_val,
                lon_val,
                event_data["route_id"],
                event_data["event_type"],
                event_data["severity"],
                float(event_data["confidence"]),
                event_data["source_modality"],
                event_data.get("software_version", "bussense-2.4.0"),
                event_data.get("road_segment_id"),
                json.dumps(event_data),
                now_str
            ))
            conn.commit()
            return True, event_data

    def upsert_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inserts or replaces an event in the database, updating coordinates, media, and payload.
        """
        event_id = event_data["event_id"]
        now_str = datetime.now(timezone.utc).isoformat()
        lat_val = float(event_data["gps_lat"]) if event_data.get("gps_lat") is not None else None
        lon_val = float(event_data["gps_lon"]) if event_data.get("gps_lon") is not None else None
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO events (
                event_id, bus_id, timestamp, gps_lat, gps_lon, route_id,
                event_type, severity, confidence, source_modality, software_version,
                road_segment_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                event_data["bus_id"],
                float(event_data["timestamp"]),
                lat_val,
                lon_val,
                event_data["route_id"],
                event_data["event_type"],
                event_data["severity"],
                float(event_data["confidence"]),
                event_data["source_modality"],
                event_data.get("software_version", "bussense-2.4.0"),
                event_data.get("road_segment_id"),
                json.dumps(event_data),
                now_str
            ))
            conn.commit()
            return event_data

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            # 1. Direct match in events table
            cur.execute("SELECT payload_json FROM events WHERE event_id = ?", (event_id,))
            row = cur.fetchone()
            if row:
                return json.loads(row["payload_json"])

            # 2. Try stripped INFRA- prefix if present
            if event_id.startswith("INFRA-"):
                clean_id = event_id[6:]
                cur.execute("SELECT payload_json FROM events WHERE event_id = ?", (clean_id,))
                row = cur.fetchone()
                if row:
                    return json.loads(row["payload_json"])

            # 3. Try with INFRA- prefix if not present
            cur.execute("SELECT payload_json FROM events WHERE event_id = ?", (f"INFRA-{event_id}",))
            row = cur.fetchone()
            if row:
                return json.loads(row["payload_json"])

            # 4. Fallback: check infrastructure_events table
            cur.execute("SELECT * FROM infrastructure_events WHERE event_id = ?", (event_id,))
            row = cur.fetchone()
            if not row and event_id.startswith("INFRA-"):
                cur.execute("SELECT * FROM infrastructure_events WHERE event_id = ?", (event_id[6:],))
                row = cur.fetchone()
            if not row and not event_id.startswith("INFRA-"):
                cur.execute("SELECT * FROM infrastructure_events WHERE event_id = ?", (f"INFRA-{event_id}",))
                row = cur.fetchone()

            if row:
                d = dict(row)
                params = json.loads(d.pop("parameters_json") or "{}")
                refs = json.loads(d.pop("evidence_refs_json") or "[]")
                img_url = next((r["url"] for r in refs if r.get("type") in ["image", "frame"]), None) or params.get("image") or params.get("evidence_image_url") or params.get("best_frame")
                clip_url = next((r["url"] for r in refs if r.get("type") in ["video", "clip"]), None) or params.get("event_clip")
                bus_ids = json.loads(d.pop("bus_ids_json") or '["BUS_001"]')
                
                res = {
                    "event_id": d["event_id"],
                    "event_type": d["event_type"],
                    "road_segment_id": d.get("road_segment_id"),
                    "gps_lat": d.get("latitude"),
                    "gps_lon": d.get("longitude"),
                    "heading": d.get("heading"),
                    "severity": params.get("severity", "HIGH"),
                    "confidence": float(d.get("temporal_consistency") or params.get("confidence", 0.92)),
                    "bus_id": bus_ids[0] if bus_ids else "BUS_001",
                    "timestamp": d.get("last_seen", time.time()),
                    "source_modality": params.get("source_modality", "CAM_STEREO_FUSION"),
                    "status": d.get("status"),
                    "chainage_m": params.get("chainage_m", 0.0),
                    "image": img_url,
                    "best_frame": img_url,
                    "evidence_image_url": img_url,
                    "event_clip": clip_url,
                    "parameters": params,
                    **params
                }
                return res

            return None

    def list_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        bus_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        query = "SELECT payload_json FROM events WHERE 1=1"
        params: List[Any] = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if bus_id:
            query += " AND bus_id = ?"
            params.append(bus_id)
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            return [json.loads(row["payload_json"]) for row in cur.fetchall()]

    # =========================================================================
    # HEARTBEATS
    # =========================================================================

    def record_heartbeat(self, heartbeat: Dict[str, Any]) -> int:
        now_str = datetime.now(timezone.utc).isoformat()
        lat_val = float(heartbeat["gps_lat"]) if heartbeat.get("gps_lat") is not None else None
        lon_val = float(heartbeat["gps_lon"]) if heartbeat.get("gps_lon") is not None else None
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO bus_heartbeats (
                bus_id, timestamp, gps_lat, gps_lon, speed_kmh, system_status, payload_json, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                heartbeat["bus_id"],
                float(heartbeat["timestamp"]),
                lat_val,
                lon_val,
                float(heartbeat["speed_kmh"]),
                heartbeat.get("system_status", "OK"),
                json.dumps(heartbeat),
                now_str
            ))
            conn.commit()
            return cur.lastrowid

    # =========================================================================
    # MEDIA STORAGE
    # =========================================================================

    def save_media_file(
        self,
        event_id: str,
        bus_id: str,
        filename: str,
        content_bytes: bytes,
        media_type: str = "image",
        mime_type: str = "image/jpeg"
    ) -> Dict[str, Any]:
        """
        Saves uploaded media bytes to deterministic path:
        outputs/server_media/events/<bus_id>/<date>/<event_id>/<filename>
        """
        root_dir = Path(self.db_path).parent.parent
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_dir = root_dir / "server_media" / "events" / bus_id / date_str / event_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_name = os.path.basename(filename).replace(" ", "_")
        target_file = target_dir / safe_name

        with open(target_file, "wb") as f:
            f.write(content_bytes)

        checksum = hashlib.sha256(content_bytes).hexdigest()
        media_id = f"MED_{checksum[:12]}"
        now_str = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO media_records (
                media_id, event_id, bus_id, media_type, mime_type, filename,
                size_bytes, checksum_sha256, file_path, uploaded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                media_id, event_id, bus_id, media_type, mime_type, safe_name,
                len(content_bytes), checksum, str(target_file), now_str
            ))
            conn.commit()

        return {
            "media_id": media_id,
            "event_id": event_id,
            "filename": safe_name,
            "size_bytes": len(content_bytes),
            "mime_type": mime_type,
            "checksum_sha256": checksum,
            "file_path": str(target_file)
        }

    def get_media(self, media_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM media_records WHERE media_id = ?", (media_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_media_for_event(self, event_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM media_records WHERE event_id = ? ORDER BY uploaded_at ASC", (event_id,))
            return [dict(row) for row in cur.fetchall()]

    # =========================================================================
    # SUMMARY / HEALTH STATS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM events")
            total_events = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM bus_heartbeats")
            total_heartbeats = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM media_records")
            total_media = cur.fetchone()[0]
            cur.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
            breakdown = dict(cur.fetchall())

            return {
                "total_events": total_events,
                "total_heartbeats": total_heartbeats,
                "total_media": total_media,
                "event_breakdown": breakdown,
                "database_path": self.db_path
            }

    # =========================================================================
    # CSR PARTNERS & ROAD REPAIR SPONSORSHIP
    # =========================================================================

    def _seed_default_csr_data(self, cur: sqlite3.Cursor):
        """
        Seeds initial realistic CSR partners and demo request history if empty.
        """
        cur.execute("SELECT COUNT(*) FROM csr_partners")
        if cur.fetchone()[0] == 0:
            now_str = datetime.now(timezone.utc).isoformat()
            partners = [
                (
                    "TVS_MOBILITY_CSR",
                    "TVS Mobility Safety Foundation (DEMO CSR PARTNER)",
                    "🏍️",
                    7500000.0,
                    2250000.0,
                    json.dumps(["Road Safety", "Two-Wheeler Accident Prevention", "Transit Corridors", "Pothole Elimination"]),
                    "Tiruchirappalli & Central Tamil Nadu Corridors",
                    "Anand Ramanathan (CSR Road Safety Director)",
                    "demo.csr@tvs-mobility-demo.in",
                    "+91 44 2833 0000",
                    "https://www.tvsmotor.com/csr",
                    1,
                    now_str
                ),
                (
                    "LNT_INFRA_CSR",
                    "L&T Construction & Urban Infra CSR (DEMO CSR PARTNER)",
                    "🏗️",
                    12000000.0,
                    4500000.0,
                    json.dumps(["Heavy Transit Pavement", "Urban Infrastructure", "Highway Grade Restoration", "Pedestrian Crossings"]),
                    "Tamil Nadu National Highway & City Arterials",
                    "Rajeshwari Sundaram (Urban Infrastructure Grants)",
                    "demo.grants@lnt-infra-demo.in",
                    "+91 44 2252 6000",
                    "https://www.larsentoubro.com/corporate/sustainability",
                    1,
                    now_str
                ),
                (
                    "BHEL_TRICHY_CSR",
                    "BHEL Trichy Community Welfare Trust (DEMO CSR PARTNER)",
                    "⚙️",
                    3500000.0,
                    850000.0,
                    json.dumps(["Local Trichy Municipal Roads", "Public Bus Route Repair", "Civic Safety", "Industrial Zone Roads"]),
                    "Tiruchirappalli Metro (Cantonment, Thillai Nagar, Kailasapuram)",
                    "K. Vijayaraghavan (Community Safety Liaison)",
                    "demo.csr@bhel-trichy-demo.in",
                    "+91 431 257 0000",
                    "https://www.bhel.com/csr",
                    1,
                    now_str
                ),
                (
                    "DALMIA_ROADS_CSR",
                    "Dalmia Bharat Foundation — Clean & Safe Roads (DEMO CSR PARTNER)",
                    "🌿",
                    5000000.0,
                    1400000.0,
                    json.dumps(["Eco-friendly Pavement Concrete", "Sustainable Road Patching", "Rural-Urban Link Roads"]),
                    "Central Tamil Nadu (Trichy, Ariyalur, Dalmiapuram)",
                    "Meenakshi Natarajan (CSR Infrastructure Specialist)",
                    "demo.csr@dalmia-foundation-demo.in",
                    "+91 44 2841 0000",
                    "https://www.dalmiabharat.com/csr",
                    1,
                    now_str
                )
            ]
            cur.executemany("""
            INSERT INTO csr_partners (
                company_id, company_name, logo, csr_budget_inr, available_budget_inr,
                focus_areas, geography, contact_name, contact_email, contact_phone,
                website, demo_flag, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, partners)

        cur.execute("SELECT COUNT(*) FROM csr_requests")
        if cur.fetchone()[0] == 0:
            now_str = datetime.now(timezone.utc).isoformat()
            requests = [
                (
                    "CSR-REQ-2026-001",
                    "BHEL_TRICHY_CSR",
                    "EVT-POT-004",
                    "BUS-01",
                    1773998240.0,
                    10.8122,
                    78.6895,
                    4.2,
                    0.22,
                    "HIGH",
                    "Vestry Roundabout / Collectorate Rd",
                    6450.0,
                    6450.0,
                    "http://localhost:8000/api/media/MED_b0ec22998a4d",
                    "Priority arterial junction with heavy school bus and civic transit flow.",
                    "UNDER REVIEW",
                    now_str,
                    now_str
                ),
                (
                    "CSR-REQ-2026-002",
                    "TVS_MOBILITY_CSR",
                    "EVT-POT-005",
                    "BUS-01",
                    1773998260.0,
                    10.8340,
                    78.7070,
                    4.8,
                    0.28,
                    "CRITICAL",
                    "Cauvery River Bridge Approach",
                    7900.0,
                    7900.0,
                    "http://localhost:8000/api/media/MED_27c592a8e801",
                    "Critical bridge approach cavity endangering two-wheelers and night commuter buses.",
                    "APPROVED",
                    now_str,
                    now_str
                )
            ]
            cur.executemany("""
            INSERT INTO csr_requests (
                request_id, company_id, pothole_event_id, bus_id, timestamp,
                gps_lat, gps_lon, depth_cm, area_m2, severity, road_name,
                estimated_repair_cost_inr, requested_amount_inr, evidence_image,
                notes, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, requests)

    def get_csr_partners(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM csr_partners ORDER BY available_budget_inr DESC")
            results = []
            for row in cur.fetchall():
                d = dict(row)
                try:
                    d["focus_areas"] = json.loads(d["focus_areas"])
                except Exception:
                    pass
                d["demo_flag"] = bool(d.get("demo_flag", 1))
                results.append(d)
            return results

    def get_csr_partner(self, company_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM csr_partners WHERE company_id = ?", (company_id,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["focus_areas"] = json.loads(d["focus_areas"])
            except Exception:
                pass
            d["demo_flag"] = bool(d.get("demo_flag", 1))
            return d

    def create_csr_partner(self, data: Dict[str, Any]) -> Dict[str, Any]:
        now_str = datetime.now(timezone.utc).isoformat()
        focus_areas = data.get("focus_areas", [])
        if isinstance(focus_areas, list):
            focus_str = json.dumps(focus_areas)
        else:
            focus_str = str(focus_areas)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO csr_partners (
                company_id, company_name, logo, csr_budget_inr, available_budget_inr,
                focus_areas, geography, contact_name, contact_email, contact_phone,
                website, demo_flag, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["company_id"],
                data["company_name"],
                data.get("logo", "🏢"),
                float(data["csr_budget_inr"]),
                float(data.get("available_budget_inr", data["csr_budget_inr"])),
                focus_str,
                data.get("geography", "Tiruchirappalli & Tamil Nadu"),
                data.get("contact_name", "CSR Coordinator"),
                data.get("contact_email", "csr@demo.in"),
                data.get("contact_phone", "+91 44 0000 0000"),
                data.get("website", ""),
                1 if data.get("demo_flag", True) else 0,
                now_str
            ))
            conn.commit()
        return self.get_csr_partner(data["company_id"])

    def get_csr_requests(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            query = """
            SELECT r.*, p.company_name, p.logo as company_logo
            FROM csr_requests r
            LEFT JOIN csr_partners p ON r.company_id = p.company_id
            """
            params = []
            if status_filter:
                query += " WHERE r.status = ?"
                params.append(status_filter.upper())
            query += " ORDER BY r.timestamp DESC"
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def get_csr_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            query = """
            SELECT r.*, p.company_name, p.logo as company_logo, p.contact_name, p.contact_email, p.contact_phone
            FROM csr_requests r
            LEFT JOIN csr_partners p ON r.company_id = p.company_id
            WHERE r.request_id = ?
            """
            cur.execute(query, (request_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def create_csr_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM csr_requests")
            count = cur.fetchone()[0]
            request_id = f"CSR-REQ-2026-{count + 1:03d}"
            now_ts = datetime.now(timezone.utc).timestamp()
            now_str = datetime.now(timezone.utc).isoformat()

            cur.execute("""
            INSERT INTO csr_requests (
                request_id, company_id, pothole_event_id, bus_id, timestamp,
                gps_lat, gps_lon, depth_cm, area_m2, severity, road_name,
                estimated_repair_cost_inr, requested_amount_inr, evidence_image,
                notes, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id,
                data["company_id"],
                data["pothole_event_id"],
                data.get("bus_id", "BUS-01"),
                now_ts,
                data.get("gps_lat"),
                data.get("gps_lon"),
                data.get("depth_cm"),
                data.get("area_m2"),
                data.get("severity", "MEDIUM"),
                data.get("road_name", "Tiruchirappalli Corridor"),
                float(data["estimated_repair_cost_inr"]),
                float(data["requested_amount_inr"]),
                data.get("evidence_image", ""),
                data.get("notes", ""),
                "REQUESTED",
                now_str,
                now_str
            ))

            # Deduct from company available budget
            cur.execute("""
            UPDATE csr_partners
            SET available_budget_inr = MAX(0.0, available_budget_inr - ?)
            WHERE company_id = ?
            """, (float(data["requested_amount_inr"]), data["company_id"]))

            conn.commit()
            return self.get_csr_request(request_id)

    def update_csr_request_status(self, request_id: str, new_status: str) -> Optional[Dict[str, Any]]:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            UPDATE csr_requests
            SET status = ?, updated_at = ?
            WHERE request_id = ?
            """, (new_status.upper(), now_str, request_id))
            conn.commit()
            return self.get_csr_request(request_id)

    # =========================================================================
    # 12. INFRASTRUCTURE OBSERVATIONS (Section 18)
    # =========================================================================

    def insert_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs_id = obs.get("observation_id", f"OBS-{int(time.time()*1000)}")
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO infrastructure_observations (
                observation_id, bus_id, timestamp, latitude, longitude, heading,
                road_segment_id, event_type, track_id, model_confidence,
                visibility_score, temporal_consistency, parameters_json,
                evidence_refs_json, simulation_mode, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                obs_id,
                obs.get("bus_id", "BUS-01"),
                float(obs.get("timestamp", time.time())),
                obs.get("latitude"),
                obs.get("longitude"),
                obs.get("heading"),
                obs.get("road_segment_id", "SEG-TRICHY-01"),
                obs.get("event_type", "UNKNOWN"),
                obs.get("track_id"),
                float(obs.get("model_confidence", 0.9)),
                float(obs.get("visibility_score", 0.95)),
                float(obs.get("temporal_consistency", 1.0)),
                json.dumps(obs.get("parameters", {})),
                json.dumps(obs.get("evidence_refs", [])),
                1 if obs.get("simulation_mode") else 0,
                now_str
            ))
            conn.commit()
            return self.get_observation(obs_id)

    def get_observation(self, obs_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM infrastructure_observations WHERE observation_id = ?", (obs_id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            res["parameters"] = json.loads(res.pop("parameters_json"))
            res["evidence_refs"] = json.loads(res.pop("evidence_refs_json"))
            res["simulation_mode"] = bool(res["simulation_mode"])
            return res

    def list_observations(
        self,
        bus_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            query = "SELECT * FROM infrastructure_observations WHERE 1=1"
            params: List[Any] = []
            if bus_id:
                query += " AND bus_id = ?"
                params.append(bus_id)
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            cur.execute(query, params)
            rows = cur.fetchall()
            out = []
            for r in rows:
                item = dict(r)
                item["parameters"] = json.loads(item.pop("parameters_json"))
                item["evidence_refs"] = json.loads(item.pop("evidence_refs_json"))
                item["simulation_mode"] = bool(item["simulation_mode"])
                out.append(item)
            return out

    # =========================================================================
    # 13. REFERENCE ASSETS (Section 15)
    # =========================================================================

    def insert_reference_asset(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        asset_id = asset["asset_id"]
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO reference_assets (
                asset_id, road_segment_id, asset_type, asset_subtype,
                expected_state, reference_source, source_quality,
                geometry_json, allowed_openings_json, last_verified, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                asset_id,
                asset.get("road_segment_id", "SEG-TRICHY-01"),
                asset.get("asset_type", "OTHER_INFRASTRUCTURE"),
                asset.get("asset_subtype"),
                asset.get("expected_state", "YES"),
                asset.get("reference_source", "MANUAL_DEMO_REFERENCE"),
                asset.get("source_quality", "DEMO"),
                json.dumps(asset.get("geometry", {})),
                json.dumps(asset.get("allowed_openings", [])),
                asset.get("last_verified", now_str),
                now_str
            ))
            conn.commit()
            return self.get_reference_asset(asset_id)

    def get_reference_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM reference_assets WHERE asset_id = ?", (asset_id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            res["geometry"] = json.loads(res.pop("geometry_json") or "{}")
            res["allowed_openings"] = json.loads(res.pop("allowed_openings_json") or "[]")
            return res

    def list_reference_assets(
        self,
        asset_type: Optional[str] = None,
        road_segment_id: Optional[str] = None,
        expected_state: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            query = "SELECT * FROM reference_assets WHERE 1=1"
            params: List[Any] = []
            if asset_type:
                query += " AND asset_type = ?"
                params.append(asset_type)
            if road_segment_id:
                query += " AND road_segment_id = ?"
                params.append(road_segment_id)
            if expected_state:
                query += " AND expected_state = ?"
                params.append(expected_state)
            cur.execute(query, params)
            rows = cur.fetchall()
            out = []
            for r in rows:
                item = dict(r)
                item["geometry"] = json.loads(item.pop("geometry_json") or "{}")
                item["allowed_openings"] = json.loads(item.pop("allowed_openings_json") or "[]")
                out.append(item)
            return out

    # =========================================================================
    # 14. INFRASTRUCTURE EVENTS (Section 19)
    # =========================================================================

    def insert_or_update_infra_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        event_id = ev["event_id"]
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT created_at FROM infrastructure_events WHERE event_id = ?", (event_id,))
            existing = cur.fetchone()
            created_at = existing["created_at"] if existing else now_str

            cur.execute("""
            INSERT OR REPLACE INTO infrastructure_events (
                event_id, event_type, road_segment_id, reference_asset_id,
                status, first_seen, last_seen, observation_count,
                unique_bus_count, unique_pass_count, unique_day_count,
                visibility_summary_json, temporal_consistency, spatial_consistency,
                reference_support, conflict_count, parameters_json,
                evidence_refs_json, simulation_mode, latitude, longitude, heading,
                bus_ids_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                ev["event_type"],
                ev.get("road_segment_id"),
                ev.get("reference_asset_id"),
                ev.get("status", "OBSERVED"),
                float(ev.get("first_seen", time.time())),
                float(ev.get("last_seen", time.time())),
                int(ev.get("observation_count", 1)),
                int(ev.get("unique_bus_count", 1)),
                int(ev.get("unique_pass_count", 1)),
                int(ev.get("unique_day_count", 1)),
                json.dumps(ev.get("visibility_summary", {})),
                float(ev.get("temporal_consistency", 1.0)),
                float(ev.get("spatial_consistency", 1.0)),
                ev.get("reference_support", "UNKNOWN"),
                int(ev.get("conflict_count", 0)),
                json.dumps(ev.get("parameters", {})),
                json.dumps(ev.get("evidence_refs", [])),
                1 if ev.get("simulation_mode") else 0,
                ev.get("latitude"),
                ev.get("longitude"),
                ev.get("heading"),
                json.dumps(ev.get("bus_ids", [])),
                created_at,
                now_str
            ))
            conn.commit()
            return self.get_infra_event(event_id)

    def get_infra_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM infrastructure_events WHERE event_id = ?", (event_id,))
            row = cur.fetchone()
            if not row and event_id.startswith("INFRA-"):
                cur.execute("SELECT * FROM infrastructure_events WHERE event_id = ?", (event_id[6:],))
                row = cur.fetchone()
            if not row and not event_id.startswith("INFRA-"):
                cur.execute("SELECT * FROM infrastructure_events WHERE event_id = ?", (f"INFRA-{event_id}",))
                row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            res["visibility_summary"] = json.loads(res.pop("visibility_summary_json") or "{}")
            res["parameters"] = json.loads(res.pop("parameters_json") or "{}")
            res["evidence_refs"] = json.loads(res.pop("evidence_refs_json") or "[]")
            res["bus_ids"] = json.loads(res.pop("bus_ids_json") or "[]")
            res["simulation_mode"] = bool(res["simulation_mode"])
            return res

    def list_infra_events(
        self,
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        road_segment_id: Optional[str] = None,
        simulation_mode: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            query = "SELECT * FROM infrastructure_events WHERE 1=1"
            params: List[Any] = []
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            if status:
                query += " AND status = ?"
                params.append(status)
            if road_segment_id:
                query += " AND road_segment_id = ?"
                params.append(road_segment_id)
            if simulation_mode is not None:
                query += " AND simulation_mode = ?"
                params.append(1 if simulation_mode else 0)
            query += " ORDER BY last_seen DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            cur.execute(query, params)
            rows = cur.fetchall()
            out = []
            for r in rows:
                item = dict(r)
                item["visibility_summary"] = json.loads(item.pop("visibility_summary_json") or "{}")
                item["parameters"] = json.loads(item.pop("parameters_json") or "{}")
                item["evidence_refs"] = json.loads(item.pop("evidence_refs_json") or "[]")
                item["bus_ids"] = json.loads(item.pop("bus_ids_json") or "[]")
                item["simulation_mode"] = bool(item["simulation_mode"])
                out.append(item)
            return out

    # =========================================================================
    # 15. BUS ROAD-CONSTRAINED TELEMETRY (Section 23.6)
    # =========================================================================

    def record_bus_telemetry(self, telem: Dict[str, Any]) -> int:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO bus_telemetry (
                bus_id, timestamp, latitude, longitude, heading,
                road_name, osm_way_id, road_segment_id, distance_along_road_m,
                distance_to_centerline_m, speed_kmh, simulation_mode, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                telem["bus_id"],
                float(telem["timestamp"]),
                float(telem["latitude"]),
                float(telem["longitude"]),
                float(telem["heading"]),
                telem["road_name"],
                telem.get("osm_way_id"),
                telem["road_segment_id"],
                float(telem["distance_along_road_m"]),
                float(telem["distance_to_centerline_m"]),
                float(telem.get("speed_kmh", 25.0)),
                1 if telem.get("simulation_mode", True) else 0,
                now_str
            ))
            conn.commit()
            return cur.lastrowid

    def get_bus_telemetry(self, bus_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT * FROM bus_telemetry
            WHERE bus_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """, (bus_id, limit))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def list_active_buses(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT bus_id, MAX(timestamp) as last_seen, road_name,
                   latitude, longitude, heading, distance_along_road_m,
                   distance_to_centerline_m, speed_kmh, simulation_mode
            FROM bus_telemetry
            GROUP BY bus_id
            """)
            rows = cur.fetchall()
            return [dict(r) for r in rows]


