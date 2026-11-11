"""
BusSense Server Database — SQLAlchemy / MySQL & SQLite Multi-Dialect Persistence Layer
=====================================================================================
Provides thread-safe persistence, connection pooling, duplicate event suppression (idempotency),
deterministic media resolution, and 100% backward compatibility with all existing endpoints.
Supports MySQL (default) and SQLite (automatic fallback when MySQL is unconfigured/testing).
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import time

from sqlalchemy import select, update, delete, func, text, desc, distinct
from sqlalchemy.orm import Session

from bussense_server.db_config import (
    Base,
    build_engine,
    SessionLocal,
    get_database_url,
    DB_DIALECT,
)
from bussense_server.models import (
    EventModel,
    BusHeartbeat,
    MediaRecord,
    CSRPartner,
    CSRRequest,
    ReferenceAsset,
    InfrastructureObservation,
    InfrastructureEvent,
    BusTelemetry,
)


class BusSenseDatabase:
    """
    Unified database persistence layer for BusSense.
    Uses SQLAlchemy ORM & Connection Pooling backed by MySQL (or SQLite fallback).
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the database engine and tables.
        If db_path is provided (e.g. in tests or explicit sqlite path), binds to that database.
        Otherwise, uses the configured MySQL / DB_DIALECT connection pool.
        """
        if db_path is not None:
            if db_path.startswith("sqlite") or db_path.startswith("mysql"):
                self.db_url = db_path
            else:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                self.db_url = f"sqlite:///{Path(db_path).resolve()}"
            self.db_path = db_path
            self.engine = build_engine(self.db_url)
            self._session_factory = SessionLocal if self.db_url == get_database_url() else None
        else:
            self.db_url = get_database_url()
            root_dir = Path(__file__).resolve().parent.parent
            data_dir = root_dir / "outputs" / "server_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "bussense_events.db")
            self.engine = build_engine(self.db_url)
            self._session_factory = SessionLocal

        self._init_db()

    def get_session(self) -> Session:
        """Returns a new SQLAlchemy session bound to this database engine."""
        if self._session_factory and self.engine == self._session_factory.kw["bind"]:
            return self._session_factory()
        return Session(bind=self.engine)

    def _init_db(self):
        """Creates tables and indexes if they do not exist and seeds initial CSR records."""
        Base.metadata.create_all(self.engine)
        with self.get_session() as session:
            self._seed_default_csr_data(session)

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
        with self.get_session() as session:
            existing = session.execute(
                select(EventModel.payload_json).where(EventModel.event_id == event_id)
            ).scalar_one_or_none()
            if existing:
                return False, json.loads(existing)

            now_str = datetime.now(timezone.utc).isoformat()
            lat_val = float(event_data["gps_lat"]) if event_data.get("gps_lat") is not None else None
            lon_val = float(event_data["gps_lon"]) if event_data.get("gps_lon") is not None else None

            new_ev = EventModel(
                event_id=event_id,
                bus_id=event_data["bus_id"],
                timestamp=float(event_data["timestamp"]),
                gps_lat=lat_val,
                gps_lon=lon_val,
                route_id=event_data["route_id"],
                event_type=event_data["event_type"],
                severity=event_data["severity"],
                confidence=float(event_data["confidence"]),
                source_modality=event_data["source_modality"],
                software_version=event_data.get("software_version", "bussense-2.4.0"),
                road_segment_id=event_data.get("road_segment_id"),
                payload_json=json.dumps(event_data),
                created_at=now_str,
            )
            session.add(new_ev)
            session.commit()
            return True, event_data

    def upsert_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inserts or updates an event in the database, updating coordinates, media, and payload.
        """
        event_id = event_data["event_id"]
        now_str = datetime.now(timezone.utc).isoformat()
        lat_val = float(event_data["gps_lat"]) if event_data.get("gps_lat") is not None else None
        lon_val = float(event_data["gps_lon"]) if event_data.get("gps_lon") is not None else None

        with self.get_session() as session:
            ev = session.get(EventModel, event_id)
            if ev:
                ev.bus_id = event_data["bus_id"]
                ev.timestamp = float(event_data["timestamp"])
                ev.gps_lat = lat_val
                ev.gps_lon = lon_val
                ev.route_id = event_data["route_id"]
                ev.event_type = event_data["event_type"]
                ev.severity = event_data["severity"]
                ev.confidence = float(event_data["confidence"])
                ev.source_modality = event_data["source_modality"]
                ev.software_version = event_data.get("software_version", "bussense-2.4.0")
                ev.road_segment_id = event_data.get("road_segment_id")
                ev.payload_json = json.dumps(event_data)
                ev.created_at = now_str
            else:
                ev = EventModel(
                    event_id=event_id,
                    bus_id=event_data["bus_id"],
                    timestamp=float(event_data["timestamp"]),
                    gps_lat=lat_val,
                    gps_lon=lon_val,
                    route_id=event_data["route_id"],
                    event_type=event_data["event_type"],
                    severity=event_data["severity"],
                    confidence=float(event_data["confidence"]),
                    source_modality=event_data["source_modality"],
                    software_version=event_data.get("software_version", "bussense-2.4.0"),
                    road_segment_id=event_data.get("road_segment_id"),
                    payload_json=json.dumps(event_data),
                    created_at=now_str,
                )
                session.add(ev)
            session.commit()
            return event_data

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            # 1. Direct match in events table
            row = session.execute(
                select(EventModel.payload_json).where(EventModel.event_id == event_id)
            ).scalar_one_or_none()
            if row:
                return json.loads(row)

            # 2. Try stripped INFRA- prefix if present
            if event_id.startswith("INFRA-"):
                clean_id = event_id[6:]
                row = session.execute(
                    select(EventModel.payload_json).where(EventModel.event_id == clean_id)
                ).scalar_one_or_none()
                if row:
                    return json.loads(row)

            # 3. Try with INFRA- prefix if not present
            row = session.execute(
                select(EventModel.payload_json).where(EventModel.event_id == f"INFRA-{event_id}")
            ).scalar_one_or_none()
            if row:
                return json.loads(row)

            # 4. Fallback: check infrastructure_events table
            infra_row = session.get(InfrastructureEvent, event_id)
            if not infra_row and not event_id.startswith("INFRA-"):
                infra_row = session.get(InfrastructureEvent, f"INFRA-{event_id}")

            if infra_row:
                params = json.loads(infra_row.parameters_json or "{}")
                evidence = json.loads(infra_row.evidence_refs_json or "[]")
                img_url = next((e["url"] for e in evidence if e.get("type") == "image"), None)
                clip_url = next((e["url"] for e in evidence if e.get("type") == "video"), None)
                bus_ids = json.loads(infra_row.bus_ids_json or "[]")
                primary_bus = bus_ids[0] if bus_ids else "BUS-01"

                payload = {
                    "event_id": infra_row.event_id,
                    "event_type": infra_row.event_type,
                    "bus_id": primary_bus,
                    "timestamp": infra_row.last_seen,
                    "gps_lat": infra_row.latitude,
                    "gps_lon": infra_row.longitude,
                    "heading": infra_row.heading,
                    "road_segment_id": infra_row.road_segment_id,
                    "severity": params.get("severity", "MEDIUM"),
                    "confidence": infra_row.temporal_consistency,
                    "source_modality": "INFRASTRUCTURE_PIPELINE",
                    "software_version": "bussense-2.4.0",
                    "status": infra_row.status,
                    "observation_count": infra_row.observation_count,
                    "unique_bus_count": infra_row.unique_bus_count,
                    "image": img_url,
                    "best_frame": img_url,
                    "evidence_image_url": img_url,
                    "event_clip": clip_url,
                    "chainage_m": params.get("chainage_m"),
                    "demo_route_location": {
                        "road_name": params.get("road_name", "Bharathidasan Salai"),
                        "chainage_m": params.get("chainage_m"),
                        "lat": infra_row.latitude,
                        "lon": infra_row.longitude,
                        "heading": infra_row.heading,
                        "landmark": params.get("landmark", "Bharathidasan Corridor"),
                    },
                }
                payload.update(params)
                return payload

            return None

    def list_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        bus_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            stmt = select(EventModel.payload_json)
            if event_type:
                stmt = stmt.where(EventModel.event_type == event_type)
            if severity:
                stmt = stmt.where(EventModel.severity == severity)
            if bus_id:
                stmt = stmt.where(EventModel.bus_id == bus_id)

            stmt = stmt.order_by(desc(EventModel.timestamp)).limit(limit).offset(offset)
            rows = session.execute(stmt).scalars().all()
            return [json.loads(r) for r in rows]

    # =========================================================================
    # HEARTBEATS & BUS TELEMETRY
    # =========================================================================

    def record_heartbeat(self, heartbeat: Dict[str, Any]) -> int:
        now_str = datetime.now(timezone.utc).isoformat()
        lat_val = float(heartbeat["gps_lat"]) if heartbeat.get("gps_lat") is not None else None
        lon_val = float(heartbeat["gps_lon"]) if heartbeat.get("gps_lon") is not None else None

        with self.get_session() as session:
            hb = BusHeartbeat(
                bus_id=heartbeat["bus_id"],
                timestamp=float(heartbeat["timestamp"]),
                gps_lat=lat_val,
                gps_lon=lon_val,
                speed_kmh=float(heartbeat["speed_kmh"]),
                system_status=heartbeat.get("system_status", "OK"),
                payload_json=json.dumps(heartbeat),
                received_at=now_str,
            )
            session.add(hb)
            session.commit()
            return hb.id

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
        mime_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """
        Saves uploaded media bytes to deterministic path:
        outputs/server_media/events/<bus_id>/<date>/<event_id>/<filename>
        """
        root_dir = Path(__file__).resolve().parent.parent / "outputs"
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_dir = root_dir / "server_media" / "events" / bus_id / date_str / event_id
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_name = os.path.basename(filename).replace(" ", "_")
        target_file = target_dir / safe_name

        with open(target_file, "wb") as f:
            f.write(content_bytes)

        checksum = hashlib.sha256(content_bytes).hexdigest()
        media_id = f"MED_{checksum[:12]}"
        now_str = datetime.now(timezone.utc).isoformat()

        with self.get_session() as session:
            rec = session.get(MediaRecord, media_id)
            if rec:
                rec.event_id = event_id
                rec.bus_id = bus_id
                rec.media_type = media_type
                rec.mime_type = mime_type
                rec.filename = safe_name
                rec.size_bytes = len(content_bytes)
                rec.checksum_sha256 = checksum
                rec.file_path = str(target_file)
                rec.uploaded_at = now_str
            else:
                rec = MediaRecord(
                    media_id=media_id,
                    event_id=event_id,
                    bus_id=bus_id,
                    media_type=media_type,
                    mime_type=mime_type,
                    filename=safe_name,
                    size_bytes=len(content_bytes),
                    checksum_sha256=checksum,
                    file_path=str(target_file),
                    uploaded_at=now_str,
                )
                session.add(rec)
            session.commit()

        return {
            "media_id": media_id,
            "event_id": event_id,
            "filename": safe_name,
            "size_bytes": len(content_bytes),
            "mime_type": mime_type,
            "checksum_sha256": checksum,
            "file_path": str(target_file),
        }

    def get_media(self, media_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            rec = session.get(MediaRecord, media_id)
            if not rec:
                return None
            return {
                "media_id": rec.media_id,
                "event_id": rec.event_id,
                "bus_id": rec.bus_id,
                "media_type": rec.media_type,
                "mime_type": rec.mime_type,
                "filename": rec.filename,
                "size_bytes": rec.size_bytes,
                "checksum_sha256": rec.checksum_sha256,
                "file_path": rec.file_path,
                "uploaded_at": rec.uploaded_at,
            }

    def get_media_for_event(self, event_id: str) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            stmt = select(MediaRecord).where(MediaRecord.event_id == event_id).order_by(MediaRecord.uploaded_at.asc())
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "media_id": r.media_id,
                    "event_id": r.event_id,
                    "bus_id": r.bus_id,
                    "media_type": r.media_type,
                    "mime_type": r.mime_type,
                    "filename": r.filename,
                    "size_bytes": r.size_bytes,
                    "checksum_sha256": r.checksum_sha256,
                    "file_path": r.file_path,
                    "uploaded_at": r.uploaded_at,
                }
                for r in rows
            ]

    # =========================================================================
    # SUMMARY / HEALTH STATS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        with self.get_session() as session:
            total_events = session.execute(select(func.count(EventModel.event_id))).scalar() or 0
            total_heartbeats = session.execute(select(func.count(BusHeartbeat.id))).scalar() or 0
            total_media = session.execute(select(func.count(MediaRecord.media_id))).scalar() or 0

            breakdown_query = session.execute(
                select(EventModel.event_type, func.count(EventModel.event_id)).group_by(EventModel.event_type)
            ).all()
            breakdown = {row[0]: row[1] for row in breakdown_query}

            return {
                "total_events": total_events,
                "total_heartbeats": total_heartbeats,
                "total_media": total_media,
                "event_breakdown": breakdown,
                "database_path": self.db_path,
            }

    # =========================================================================
    # CSR PARTNERS & ROAD REPAIR SPONSORSHIP
    # =========================================================================

    def _seed_default_csr_data(self, session: Session):
        """Seeds initial realistic CSR partners and demo request history if empty."""
        count = session.execute(select(func.count(CSRPartner.company_id))).scalar() or 0
        if count == 0:
            now_str = datetime.now(timezone.utc).isoformat()
            partners = [
                CSRPartner(
                    company_id="TVS_MOBILITY_CSR",
                    company_name="TVS Mobility Safety Foundation (DEMO CSR PARTNER)",
                    logo="🏢",
                    csr_budget_inr=7500000.0,
                    available_budget_inr=2250000.0,
                    focus_areas=json.dumps(["Road Safety", "Two-Wheeler Accident Prevention", "Transit Corridors", "Pothole Elimination"]),
                    geography="Tiruchirappalli & Central Tamil Nadu Corridors",
                    contact_name="Anand Ramanathan (CSR Road Safety Director)",
                    contact_email="demo.csr@tvs-mobility-demo.in",
                    contact_phone="+91 44 2833 0000",
                    website="https://www.tvsmotor.com/csr",
                    demo_flag=1,
                    created_at=now_str,
                ),
                CSRPartner(
                    company_id="LNT_INFRA_CSR",
                    company_name="L&T Construction & Urban Infra CSR (DEMO CSR PARTNER)",
                    logo="🏗️",
                    csr_budget_inr=12000000.0,
                    available_budget_inr=4500000.0,
                    focus_areas=json.dumps(["Heavy Transit Pavement", "Urban Infrastructure", "Highway Grade Restoration", "Pedestrian Crossings"]),
                    geography="Tamil Nadu National Highway & City Arterials",
                    contact_name="Rajeshwari Sundaram (Urban Infrastructure Grants)",
                    contact_email="demo.grants@lnt-infra-demo.in",
                    contact_phone="+91 44 2252 6000",
                    website="https://www.larsentoubro.com/corporate/sustainability",
                    demo_flag=1,
                    created_at=now_str,
                ),
                CSRPartner(
                    company_id="BHEL_TRICHY_CSR",
                    company_name="BHEL Trichy Community Welfare Trust (DEMO CSR PARTNER)",
                    logo="⚡",
                    csr_budget_inr=3500000.0,
                    available_budget_inr=850000.0,
                    focus_areas=json.dumps(["Local Trichy Municipal Roads", "Public Bus Route Repair", "Civic Safety", "Industrial Zone Roads"]),
                    geography="Tiruchirappalli Metro (Cantonment, Thillai Nagar, Kailasapuram)",
                    contact_name="K. Vijayaraghavan (Community Safety Liaison)",
                    contact_email="demo.csr@bhel-trichy-demo.in",
                    contact_phone="+91 431 257 0000",
                    website="https://www.bhel.com/csr",
                    demo_flag=1,
                    created_at=now_str,
                ),
                CSRPartner(
                    company_id="DALMIA_ROADS_CSR",
                    company_name="Dalmia Bharat Foundation — Clean & Safe Roads (DEMO CSR PARTNER)",
                    logo="🌿",
                    csr_budget_inr=5000000.0,
                    available_budget_inr=1400000.0,
                    focus_areas=json.dumps(["Eco-friendly Pavement Concrete", "Sustainable Road Patching", "Rural-Urban Link Roads"]),
                    geography="Central Tamil Nadu (Trichy, Ariyalur, Dalmiapuram)",
                    contact_name="Meenakshi Natarajan (CSR Infrastructure Specialist)",
                    contact_email="demo.csr@dalmia-foundation-demo.in",
                    contact_phone="+91 44 2841 0000",
                    website="https://www.dalmiabharat.com/csr",
                    demo_flag=1,
                    created_at=now_str,
                ),
            ]
            session.add_all(partners)
            session.commit()

        req_count = session.execute(select(func.count(CSRRequest.request_id))).scalar() or 0
        if req_count == 0:
            now_str = datetime.now(timezone.utc).isoformat()
            demo_requests = [
                CSRRequest(
                    request_id="CSR-REQ-2026-001",
                    company_id="BHEL_TRICHY_CSR",
                    pothole_event_id="EVT-POT-004",
                    bus_id="BUS-01",
                    timestamp=1773998240.0,
                    gps_lat=10.8122,
                    gps_lon=78.6895,
                    depth_cm=4.2,
                    area_m2=0.22,
                    severity="HIGH",
                    road_name="Vestry Roundabout / Collectorate Rd",
                    estimated_repair_cost_inr=6450.0,
                    requested_amount_inr=6450.0,
                    evidence_image="http://localhost:8000/api/media/MED_b0ec22998a4d",
                    notes="Priority arterial junction with heavy school bus and civic transit flow.",
                    status="UNDER REVIEW",
                    created_at=now_str,
                    updated_at=now_str,
                ),
                CSRRequest(
                    request_id="CSR-REQ-2026-002",
                    company_id="TVS_MOBILITY_CSR",
                    pothole_event_id="EVT-POT-005",
                    bus_id="BUS-01",
                    timestamp=1773998260.0,
                    gps_lat=10.8340,
                    gps_lon=78.7070,
                    depth_cm=4.8,
                    area_m2=0.28,
                    severity="CRITICAL",
                    road_name="Cauvery River Bridge Approach",
                    estimated_repair_cost_inr=8200.0,
                    requested_amount_inr=8200.0,
                    evidence_image="http://localhost:8000/api/media/MED_b0ec22998a4d",
                    notes="High-speed bridge approach cavity creating severe two-wheeler stability hazard.",
                    status="FUNDED",
                    created_at=now_str,
                    updated_at=now_str,
                ),
                CSRRequest(
                    request_id="CSR-REQ-2026-003",
                    company_id="LNT_INFRA_CSR",
                    pothole_event_id="EVT-POT-002",
                    bus_id="BUS-01",
                    timestamp=1773998280.0,
                    gps_lat=10.8095,
                    gps_lon=78.6872,
                    depth_cm=3.8,
                    area_m2=0.18,
                    severity="HIGH",
                    road_name="Court Junction / Bharathidasan Salai",
                    estimated_repair_cost_inr=5200.0,
                    requested_amount_inr=5200.0,
                    evidence_image="http://localhost:8000/api/media/MED_b0ec22998a4d",
                    notes="District court arterial lane with continuous passenger bus transit.",
                    status="IN REPAIR",
                    created_at=now_str,
                    updated_at=now_str,
                ),
                CSRRequest(
                    request_id="CSR-REQ-2026-004",
                    company_id="DALMIA_ROADS_CSR",
                    pothole_event_id="EVT-POT-001",
                    bus_id="BUS-01",
                    timestamp=1773998300.0,
                    gps_lat=10.8065,
                    gps_lon=78.6848,
                    depth_cm=3.2,
                    area_m2=0.14,
                    severity="MEDIUM",
                    road_name="Promenade Road / Cantonment",
                    estimated_repair_cost_inr=4100.0,
                    requested_amount_inr=4100.0,
                    evidence_image="http://localhost:8000/api/media/MED_b0ec22998a4d",
                    notes="Sub-base cavitation patched with eco-concrete blend.",
                    status="REPAIRED",
                    created_at=now_str,
                    updated_at=now_str,
                ),
                CSRRequest(
                    request_id="CSR-REQ-2026-005",
                    company_id="BHEL_TRICHY_CSR",
                    pothole_event_id="EVT-POT-003",
                    bus_id="BUS-01",
                    timestamp=1773998320.0,
                    gps_lat=10.8180,
                    gps_lon=78.6942,
                    depth_cm=3.5,
                    area_m2=0.16,
                    severity="HIGH",
                    road_name="Shastri Road / Anna Nagar Signal",
                    estimated_repair_cost_inr=4800.0,
                    requested_amount_inr=4800.0,
                    evidence_image="http://localhost:8000/api/media/MED_b0ec22998a4d",
                    notes="Congested commercial transit signal requiring rapid asphalt infill.",
                    status="SUBMITTED",
                    created_at=now_str,
                    updated_at=now_str,
                ),
                CSRRequest(
                    request_id="CSR-REQ-2026-006",
                    company_id="TVS_MOBILITY_CSR",
                    pothole_event_id="EVT-POT-006",
                    bus_id="BUS-01",
                    timestamp=1773998340.0,
                    gps_lat=10.8380,
                    gps_lon=78.7100,
                    depth_cm=2.8,
                    area_m2=0.12,
                    severity="LOW",
                    road_name="Mambazhasalai / Srirangam Approach",
                    estimated_repair_cost_inr=3600.0,
                    requested_amount_inr=3600.0,
                    evidence_image="http://localhost:8000/api/media/MED_b0ec22998a4d",
                    notes="Early-stage surface fracture monitored prior to monsoon expansion.",
                    status="REQUESTED",
                    created_at=now_str,
                    updated_at=now_str,
                ),
            ]
            session.add_all(demo_requests)
            session.commit()

    def get_csr_partners(self) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            stmt = select(CSRPartner).order_by(CSRPartner.available_budget_inr.desc())
            rows = session.execute(stmt).scalars().all()
            results = []
            for p in rows:
                focus = p.focus_areas
                try:
                    focus = json.loads(p.focus_areas)
                except Exception:
                    pass
                results.append({
                    "company_id": p.company_id,
                    "company_name": p.company_name,
                    "logo": p.logo,
                    "csr_budget_inr": p.csr_budget_inr,
                    "available_budget_inr": p.available_budget_inr,
                    "focus_areas": focus,
                    "geography": p.geography,
                    "contact_name": p.contact_name,
                    "contact_email": p.contact_email,
                    "contact_phone": p.contact_phone,
                    "website": p.website,
                    "demo_flag": bool(p.demo_flag),
                    "created_at": p.created_at,
                })
            return results

    def get_csr_partner(self, company_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            p = session.get(CSRPartner, company_id)
            if not p:
                return None
            focus = p.focus_areas
            try:
                focus = json.loads(p.focus_areas)
            except Exception:
                pass
            return {
                "company_id": p.company_id,
                "company_name": p.company_name,
                "logo": p.logo,
                "csr_budget_inr": p.csr_budget_inr,
                "available_budget_inr": p.available_budget_inr,
                "focus_areas": focus,
                "geography": p.geography,
                "contact_name": p.contact_name,
                "contact_email": p.contact_email,
                "contact_phone": p.contact_phone,
                "website": p.website,
                "demo_flag": bool(p.demo_flag),
                "created_at": p.created_at,
            }

    def create_csr_partner(self, data: Dict[str, Any]) -> Dict[str, Any]:
        now_str = datetime.now(timezone.utc).isoformat()
        focus_areas = data.get("focus_areas", [])
        focus_str = json.dumps(focus_areas) if isinstance(focus_areas, list) else str(focus_areas)

        with self.get_session() as session:
            p = session.get(CSRPartner, data["company_id"])
            if p:
                p.company_name = data["company_name"]
                p.logo = data.get("logo", "🏢")
                p.csr_budget_inr = float(data["csr_budget_inr"])
                p.available_budget_inr = float(data.get("available_budget_inr", data["csr_budget_inr"]))
                p.focus_areas = focus_str
                p.geography = data.get("geography", "Tiruchirappalli & Tamil Nadu")
                p.contact_name = data.get("contact_name", "CSR Coordinator")
                p.contact_email = data.get("contact_email", "csr@demo.in")
                p.contact_phone = data.get("contact_phone", "+91 44 0000 0000")
                p.website = data.get("website", "")
                p.demo_flag = 1 if data.get("demo_flag", True) else 0
                p.created_at = now_str
            else:
                p = CSRPartner(
                    company_id=data["company_id"],
                    company_name=data["company_name"],
                    logo=data.get("logo", "🏢"),
                    csr_budget_inr=float(data["csr_budget_inr"]),
                    available_budget_inr=float(data.get("available_budget_inr", data["csr_budget_inr"])),
                    focus_areas=focus_str,
                    geography=data.get("geography", "Tiruchirappalli & Tamil Nadu"),
                    contact_name=data.get("contact_name", "CSR Coordinator"),
                    contact_email=data.get("contact_email", "csr@demo.in"),
                    contact_phone=data.get("contact_phone", "+91 44 0000 0000"),
                    website=data.get("website", ""),
                    demo_flag=1 if data.get("demo_flag", True) else 0,
                    created_at=now_str,
                )
                session.add(p)
            session.commit()
        return self.get_csr_partner(data["company_id"])

    def get_csr_requests(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            stmt = (
                select(CSRRequest, CSRPartner.company_name, CSRPartner.logo.label("company_logo"))
                .join(CSRPartner, CSRRequest.company_id == CSRPartner.company_id, isouter=True)
            )
            if status_filter:
                stmt = stmt.where(CSRRequest.status == status_filter.upper())
            stmt = stmt.order_by(desc(CSRRequest.timestamp))

            results = []
            for r, c_name, c_logo in session.execute(stmt).all():
                d = {
                    "request_id": r.request_id,
                    "company_id": r.company_id,
                    "pothole_event_id": r.pothole_event_id,
                    "bus_id": r.bus_id,
                    "timestamp": r.timestamp,
                    "gps_lat": r.gps_lat,
                    "gps_lon": r.gps_lon,
                    "depth_cm": r.depth_cm,
                    "area_m2": r.area_m2,
                    "severity": r.severity,
                    "road_name": r.road_name,
                    "estimated_repair_cost_inr": r.estimated_repair_cost_inr,
                    "requested_amount_inr": r.requested_amount_inr,
                    "evidence_image": r.evidence_image,
                    "notes": r.notes,
                    "status": r.status,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at,
                    "company_name": c_name,
                    "company_logo": c_logo,
                }
                results.append(d)
            return results

    def get_csr_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            stmt = (
                select(
                    CSRRequest,
                    CSRPartner.company_name,
                    CSRPartner.logo.label("company_logo"),
                    CSRPartner.contact_name,
                    CSRPartner.contact_email,
                    CSRPartner.contact_phone,
                )
                .join(CSRPartner, CSRRequest.company_id == CSRPartner.company_id, isouter=True)
                .where(CSRRequest.request_id == request_id)
            )
            res = session.execute(stmt).first()
            if not res:
                return None
            r, c_name, c_logo, c_contact, c_email, c_phone = res
            return {
                "request_id": r.request_id,
                "company_id": r.company_id,
                "pothole_event_id": r.pothole_event_id,
                "bus_id": r.bus_id,
                "timestamp": r.timestamp,
                "gps_lat": r.gps_lat,
                "gps_lon": r.gps_lon,
                "depth_cm": r.depth_cm,
                "area_m2": r.area_m2,
                "severity": r.severity,
                "road_name": r.road_name,
                "estimated_repair_cost_inr": r.estimated_repair_cost_inr,
                "requested_amount_inr": r.requested_amount_inr,
                "evidence_image": r.evidence_image,
                "notes": r.notes,
                "status": r.status,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
                "company_name": c_name,
                "company_logo": c_logo,
                "contact_name": c_contact,
                "contact_email": c_email,
                "contact_phone": c_phone,
            }

    def create_csr_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self.get_session() as session:
            count = session.execute(select(func.count(CSRRequest.request_id))).scalar() or 0
            request_id = f"CSR-REQ-2026-{count + 1:03d}"
            now_ts = datetime.now(timezone.utc).timestamp()
            now_str = datetime.now(timezone.utc).isoformat()

            req = CSRRequest(
                request_id=request_id,
                company_id=data["company_id"],
                pothole_event_id=data["pothole_event_id"],
                bus_id=data.get("bus_id", "BUS-01"),
                timestamp=now_ts,
                gps_lat=data.get("gps_lat"),
                gps_lon=data.get("gps_lon"),
                depth_cm=data.get("depth_cm"),
                area_m2=data.get("area_m2"),
                severity=data.get("severity", "MEDIUM"),
                road_name=data.get("road_name", "Tiruchirappalli Corridor"),
                estimated_repair_cost_inr=float(data["estimated_repair_cost_inr"]),
                requested_amount_inr=float(data["requested_amount_inr"]),
                evidence_image=data.get("evidence_image", ""),
                notes=data.get("notes", ""),
                status="REQUESTED",
                created_at=now_str,
                updated_at=now_str,
            )
            session.add(req)

            # Deduct from company available budget
            partner = session.get(CSRPartner, data["company_id"])
            if partner:
                partner.available_budget_inr = max(0.0, partner.available_budget_inr - float(data["requested_amount_inr"]))

            session.commit()
            return self.get_csr_request(request_id)

    def update_csr_request_status(self, request_id: str, new_status: str) -> Optional[Dict[str, Any]]:
        now_str = datetime.now(timezone.utc).isoformat()
        with self.get_session() as session:
            req = session.get(CSRRequest, request_id)
            if not req:
                return None
            req.status = new_status.upper()
            req.updated_at = now_str
            session.commit()
            return self.get_csr_request(request_id)

    # =========================================================================
    # 12. INFRASTRUCTURE OBSERVATIONS (Section 18)
    # =========================================================================

    def insert_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs_id = obs.get("observation_id", f"OBS-{int(time.time()*1000)}")
        now_str = datetime.now(timezone.utc).isoformat()
        with self.get_session() as session:
            existing = session.get(InfrastructureObservation, obs_id)
            if existing:
                existing.bus_id = obs.get("bus_id", "BUS-01")
                existing.timestamp = float(obs.get("timestamp", time.time()))
                existing.latitude = obs.get("latitude")
                existing.longitude = obs.get("longitude")
                existing.heading = obs.get("heading")
                existing.road_segment_id = obs.get("road_segment_id", "SEG-TRICHY-01")
                existing.event_type = obs.get("event_type", "UNKNOWN")
                existing.track_id = obs.get("track_id")
                existing.model_confidence = float(obs.get("model_confidence", 0.9))
                existing.visibility_score = float(obs.get("visibility_score", 0.95))
                existing.temporal_consistency = float(obs.get("temporal_consistency", 1.0))
                existing.parameters_json = json.dumps(obs.get("parameters", {}))
                existing.evidence_refs_json = json.dumps(obs.get("evidence_refs", []))
                existing.simulation_mode = 1 if obs.get("simulation_mode") else 0
                existing.created_at = now_str
            else:
                o = InfrastructureObservation(
                    observation_id=obs_id,
                    bus_id=obs.get("bus_id", "BUS-01"),
                    timestamp=float(obs.get("timestamp", time.time())),
                    latitude=obs.get("latitude"),
                    longitude=obs.get("longitude"),
                    heading=obs.get("heading"),
                    road_segment_id=obs.get("road_segment_id", "SEG-TRICHY-01"),
                    event_type=obs.get("event_type", "UNKNOWN"),
                    track_id=obs.get("track_id"),
                    model_confidence=float(obs.get("model_confidence", 0.9)),
                    visibility_score=float(obs.get("visibility_score", 0.95)),
                    temporal_consistency=float(obs.get("temporal_consistency", 1.0)),
                    parameters_json=json.dumps(obs.get("parameters", {})),
                    evidence_refs_json=json.dumps(obs.get("evidence_refs", [])),
                    simulation_mode=1 if obs.get("simulation_mode") else 0,
                    created_at=now_str,
                )
                session.add(o)
            session.commit()
            return self.get_observation(obs_id)

    def get_observation(self, obs_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            o = session.get(InfrastructureObservation, obs_id)
            if not o:
                return None
            return {
                "observation_id": o.observation_id,
                "bus_id": o.bus_id,
                "timestamp": o.timestamp,
                "latitude": o.latitude,
                "longitude": o.longitude,
                "heading": o.heading,
                "road_segment_id": o.road_segment_id,
                "event_type": o.event_type,
                "track_id": o.track_id,
                "model_confidence": o.model_confidence,
                "visibility_score": o.visibility_score,
                "temporal_consistency": o.temporal_consistency,
                "parameters": json.loads(o.parameters_json),
                "evidence_refs": json.loads(o.evidence_refs_json),
                "simulation_mode": bool(o.simulation_mode),
                "created_at": o.created_at,
            }

    def list_observations(
        self,
        bus_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            stmt = select(InfrastructureObservation)
            if bus_id:
                stmt = stmt.where(InfrastructureObservation.bus_id == bus_id)
            if event_type:
                stmt = stmt.where(InfrastructureObservation.event_type == event_type)
            stmt = stmt.order_by(desc(InfrastructureObservation.timestamp)).limit(limit).offset(offset)
            rows = session.execute(stmt).scalars().all()
            out = []
            for o in rows:
                out.append({
                    "observation_id": o.observation_id,
                    "bus_id": o.bus_id,
                    "timestamp": o.timestamp,
                    "latitude": o.latitude,
                    "longitude": o.longitude,
                    "heading": o.heading,
                    "road_segment_id": o.road_segment_id,
                    "event_type": o.event_type,
                    "track_id": o.track_id,
                    "model_confidence": o.model_confidence,
                    "visibility_score": o.visibility_score,
                    "temporal_consistency": o.temporal_consistency,
                    "parameters": json.loads(o.parameters_json),
                    "evidence_refs": json.loads(o.evidence_refs_json),
                    "simulation_mode": bool(o.simulation_mode),
                    "created_at": o.created_at,
                })
            return out

    # =========================================================================
    # 13. REFERENCE ASSETS (Section 15)
    # =========================================================================

    def insert_reference_asset(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        asset_id = asset["asset_id"]
        now_str = datetime.now(timezone.utc).isoformat()
        with self.get_session() as session:
            existing = session.get(ReferenceAsset, asset_id)
            if existing:
                existing.road_segment_id = asset.get("road_segment_id", "SEG-TRICHY-01")
                existing.asset_type = asset.get("asset_type", "OTHER_INFRASTRUCTURE")
                existing.asset_subtype = asset.get("asset_subtype")
                existing.expected_state = asset.get("expected_state", "YES")
                existing.reference_source = asset.get("reference_source", "MANUAL_DEMO_REFERENCE")
                existing.source_quality = asset.get("source_quality", "DEMO")
                existing.geometry_json = json.dumps(asset.get("geometry", {}))
                existing.allowed_openings_json = json.dumps(asset.get("allowed_openings", []))
                existing.last_verified = asset.get("last_verified", now_str)
                existing.created_at = now_str
            else:
                a = ReferenceAsset(
                    asset_id=asset_id,
                    road_segment_id=asset.get("road_segment_id", "SEG-TRICHY-01"),
                    asset_type=asset.get("asset_type", "OTHER_INFRASTRUCTURE"),
                    asset_subtype=asset.get("asset_subtype"),
                    expected_state=asset.get("expected_state", "YES"),
                    reference_source=asset.get("reference_source", "MANUAL_DEMO_REFERENCE"),
                    source_quality=asset.get("source_quality", "DEMO"),
                    geometry_json=json.dumps(asset.get("geometry", {})),
                    allowed_openings_json=json.dumps(asset.get("allowed_openings", [])),
                    last_verified=asset.get("last_verified", now_str),
                    created_at=now_str,
                )
                session.add(a)
            session.commit()
            return self.get_reference_asset(asset_id)

    def get_reference_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            a = session.get(ReferenceAsset, asset_id)
            if not a:
                return None
            return {
                "asset_id": a.asset_id,
                "road_segment_id": a.road_segment_id,
                "asset_type": a.asset_type,
                "asset_subtype": a.asset_subtype,
                "expected_state": a.expected_state,
                "reference_source": a.reference_source,
                "source_quality": a.source_quality,
                "geometry": json.loads(a.geometry_json or "{}"),
                "allowed_openings": json.loads(a.allowed_openings_json or "[]"),
                "last_verified": a.last_verified,
                "created_at": a.created_at,
            }

    def list_reference_assets(
        self,
        asset_type: Optional[str] = None,
        road_segment_id: Optional[str] = None,
        expected_state: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            stmt = select(ReferenceAsset)
            if asset_type:
                stmt = stmt.where(ReferenceAsset.asset_type == asset_type)
            if road_segment_id:
                stmt = stmt.where(ReferenceAsset.road_segment_id == road_segment_id)
            if expected_state:
                stmt = stmt.where(ReferenceAsset.expected_state == expected_state)
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "asset_id": a.asset_id,
                    "road_segment_id": a.road_segment_id,
                    "asset_type": a.asset_type,
                    "asset_subtype": a.asset_subtype,
                    "expected_state": a.expected_state,
                    "reference_source": a.reference_source,
                    "source_quality": a.source_quality,
                    "geometry": json.loads(a.geometry_json or "{}"),
                    "allowed_openings": json.loads(a.allowed_openings_json or "[]"),
                    "last_verified": a.last_verified,
                    "created_at": a.created_at,
                }
                for a in rows
            ]

    # =========================================================================
    # 14. INFRASTRUCTURE EVENTS (Section 19)
    # =========================================================================

    def insert_or_update_infra_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        event_id = ev["event_id"]
        now_str = datetime.now(timezone.utc).isoformat()
        with self.get_session() as session:
            existing = session.get(InfrastructureEvent, event_id)
            created_at = existing.created_at if existing else now_str

            if existing:
                existing.event_type = ev["event_type"]
                existing.road_segment_id = ev.get("road_segment_id")
                existing.reference_asset_id = ev.get("reference_asset_id")
                existing.status = ev.get("status", "CANDIDATE")
                existing.first_seen = float(ev.get("first_seen", time.time()))
                existing.last_seen = float(ev.get("last_seen", time.time()))
                existing.observation_count = int(ev.get("observation_count", 1))
                existing.unique_bus_count = int(ev.get("unique_bus_count", 1))
                existing.unique_pass_count = int(ev.get("unique_pass_count", 1))
                existing.unique_day_count = int(ev.get("unique_day_count", 1))
                existing.visibility_summary_json = json.dumps(ev.get("visibility_summary", {}))
                existing.temporal_consistency = float(ev.get("temporal_consistency", 1.0))
                existing.spatial_consistency = float(ev.get("spatial_consistency", 1.0))
                existing.reference_support = ev.get("reference_support", "UNKNOWN")
                existing.conflict_count = int(ev.get("conflict_count", 0))
                existing.parameters_json = json.dumps(ev.get("parameters", {}))
                existing.evidence_refs_json = json.dumps(ev.get("evidence_refs", []))
                existing.simulation_mode = 1 if ev.get("simulation_mode") else 0
                existing.latitude = ev.get("latitude")
                existing.longitude = ev.get("longitude")
                existing.heading = ev.get("heading")
                existing.bus_ids_json = json.dumps(ev.get("bus_ids", []))
                existing.created_at = created_at
                existing.updated_at = now_str
            else:
                new_e = InfrastructureEvent(
                    event_id=event_id,
                    event_type=ev["event_type"],
                    road_segment_id=ev.get("road_segment_id"),
                    reference_asset_id=ev.get("reference_asset_id"),
                    status=ev.get("status", "CANDIDATE"),
                    first_seen=float(ev.get("first_seen", time.time())),
                    last_seen=float(ev.get("last_seen", time.time())),
                    observation_count=int(ev.get("observation_count", 1)),
                    unique_bus_count=int(ev.get("unique_bus_count", 1)),
                    unique_pass_count=int(ev.get("unique_pass_count", 1)),
                    unique_day_count=int(ev.get("unique_day_count", 1)),
                    visibility_summary_json=json.dumps(ev.get("visibility_summary", {})),
                    temporal_consistency=float(ev.get("temporal_consistency", 1.0)),
                    spatial_consistency=float(ev.get("spatial_consistency", 1.0)),
                    reference_support=ev.get("reference_support", "UNKNOWN"),
                    conflict_count=int(ev.get("conflict_count", 0)),
                    parameters_json=json.dumps(ev.get("parameters", {})),
                    evidence_refs_json=json.dumps(ev.get("evidence_refs", [])),
                    simulation_mode=1 if ev.get("simulation_mode") else 0,
                    latitude=ev.get("latitude"),
                    longitude=ev.get("longitude"),
                    heading=ev.get("heading"),
                    bus_ids_json=json.dumps(ev.get("bus_ids", [])),
                    created_at=created_at,
                    updated_at=now_str,
                )
                session.add(new_e)
            session.commit()
            return self.get_infra_event(event_id)

    def get_infra_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            row = session.get(InfrastructureEvent, event_id)
            if not row and event_id.startswith("INFRA-"):
                row = session.get(InfrastructureEvent, event_id[6:])
            if not row and not event_id.startswith("INFRA-"):
                row = session.get(InfrastructureEvent, f"INFRA-{event_id}")

            if not row:
                return None
            return {
                "event_id": row.event_id,
                "event_type": row.event_type,
                "road_segment_id": row.road_segment_id,
                "reference_asset_id": row.reference_asset_id,
                "status": row.status,
                "first_seen": row.first_seen,
                "last_seen": row.last_seen,
                "observation_count": row.observation_count,
                "unique_bus_count": row.unique_bus_count,
                "unique_pass_count": row.unique_pass_count,
                "unique_day_count": row.unique_day_count,
                "visibility_summary": json.loads(row.visibility_summary_json or "{}"),
                "temporal_consistency": row.temporal_consistency,
                "spatial_consistency": row.spatial_consistency,
                "reference_support": row.reference_support,
                "conflict_count": row.conflict_count,
                "parameters": json.loads(row.parameters_json or "{}"),
                "evidence_refs": json.loads(row.evidence_refs_json or "[]"),
                "simulation_mode": bool(row.simulation_mode),
                "latitude": row.latitude,
                "longitude": row.longitude,
                "heading": row.heading,
                "bus_ids": json.loads(row.bus_ids_json or "[]"),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }

    def list_infra_events(
        self,
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        road_segment_id: Optional[str] = None,
        simulation_mode: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            stmt = select(InfrastructureEvent)
            if event_type:
                stmt = stmt.where(InfrastructureEvent.event_type == event_type)
            if status:
                stmt = stmt.where(InfrastructureEvent.status == status)
            if road_segment_id:
                stmt = stmt.where(InfrastructureEvent.road_segment_id == road_segment_id)
            if simulation_mode is not None:
                stmt = stmt.where(InfrastructureEvent.simulation_mode == (1 if simulation_mode else 0))

            stmt = stmt.order_by(desc(InfrastructureEvent.last_seen)).limit(limit).offset(offset)
            rows = session.execute(stmt).scalars().all()
            out = []
            for row in rows:
                out.append({
                    "event_id": row.event_id,
                    "event_type": row.event_type,
                    "road_segment_id": row.road_segment_id,
                    "reference_asset_id": row.reference_asset_id,
                    "status": row.status,
                    "first_seen": row.first_seen,
                    "last_seen": row.last_seen,
                    "observation_count": row.observation_count,
                    "unique_bus_count": row.unique_bus_count,
                    "unique_pass_count": row.unique_pass_count,
                    "unique_day_count": row.unique_day_count,
                    "visibility_summary": json.loads(row.visibility_summary_json or "{}"),
                    "temporal_consistency": row.temporal_consistency,
                    "spatial_consistency": row.spatial_consistency,
                    "reference_support": row.reference_support,
                    "conflict_count": row.conflict_count,
                    "parameters": json.loads(row.parameters_json or "{}"),
                    "evidence_refs": json.loads(row.evidence_refs_json or "[]"),
                    "simulation_mode": bool(row.simulation_mode),
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "heading": row.heading,
                    "bus_ids": json.loads(row.bus_ids_json or "[]"),
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                })
            return out

    # =========================================================================
    # 15. BUS ROAD-CONSTRAINED TELEMETRY (Section 23.6)
    # =========================================================================

    def record_bus_telemetry(self, telem: Dict[str, Any]) -> int:
        now_str = datetime.now(timezone.utc).isoformat()
        with self.get_session() as session:
            bt = BusTelemetry(
                bus_id=telem["bus_id"],
                timestamp=float(telem["timestamp"]),
                latitude=float(telem["latitude"]),
                longitude=float(telem["longitude"]),
                heading=float(telem["heading"]),
                road_name=telem["road_name"],
                osm_way_id=telem.get("osm_way_id"),
                road_segment_id=telem["road_segment_id"],
                distance_along_road_m=float(telem["distance_along_road_m"]),
                distance_to_centerline_m=float(telem["distance_to_centerline_m"]),
                speed_kmh=float(telem.get("speed_kmh", 25.0)),
                simulation_mode=1 if telem.get("simulation_mode", True) else 0,
                recorded_at=now_str,
            )
            session.add(bt)
            session.commit()
            return bt.telemetry_id

    def get_bus_telemetry(self, bus_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            stmt = select(BusTelemetry).where(BusTelemetry.bus_id == bus_id).order_by(desc(BusTelemetry.timestamp)).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "telemetry_id": r.telemetry_id,
                    "bus_id": r.bus_id,
                    "timestamp": r.timestamp,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "heading": r.heading,
                    "road_name": r.road_name,
                    "osm_way_id": r.osm_way_id,
                    "road_segment_id": r.road_segment_id,
                    "distance_along_road_m": r.distance_along_road_m,
                    "distance_to_centerline_m": r.distance_to_centerline_m,
                    "speed_kmh": r.speed_kmh,
                    "simulation_mode": bool(r.simulation_mode),
                    "recorded_at": r.recorded_at,
                }
                for r in rows
            ]

    def list_active_buses(self) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            # Query the latest telemetry row per unique bus_id
            subq = (
                select(
                    BusTelemetry.bus_id,
                    func.max(BusTelemetry.telemetry_id).label("max_id"),
                )
                .group_by(BusTelemetry.bus_id)
                .subquery()
            )

            stmt = (
                select(BusTelemetry)
                .join(
                    subq,
                    (BusTelemetry.bus_id == subq.c.bus_id) & (BusTelemetry.telemetry_id == subq.c.max_id),
                )
                .order_by(BusTelemetry.bus_id.asc())
            )

            rows = session.execute(stmt).scalars().all()
            results = []
            for r in rows:
                results.append({
                    "bus_id": r.bus_id,
                    "last_seen": r.timestamp,
                    "road_name": r.road_name,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "heading": r.heading,
                    "distance_along_road_m": r.distance_along_road_m,
                    "distance_to_centerline_m": r.distance_to_centerline_m,
                    "speed_kmh": r.speed_kmh,
                    "simulation_mode": r.simulation_mode,
                })
            return results
