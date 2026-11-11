"""
SQLAlchemy ORM Models for BusSense MySQL Database
==================================================
Exact 1:1 schema mapping preserving all tables, columns, constraints,
indexes, and default values from the original SQLite implementation.
"""

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship

from bussense_server.db_config import Base


class CSRPartner(Base):
    __tablename__ = "csr_partners"

    company_id = Column(String(128), primary_key=True)
    company_name = Column(String(255), nullable=False)
    logo = Column(Text, nullable=True)
    csr_budget_inr = Column(Float, nullable=False)
    available_budget_inr = Column(Float, nullable=False)
    focus_areas = Column(Text, nullable=False)
    geography = Column(Text, nullable=False)
    contact_name = Column(String(255), nullable=False)
    contact_email = Column(String(255), nullable=False)
    contact_phone = Column(String(64), nullable=False)
    website = Column(String(255), nullable=True)
    demo_flag = Column(Integer, default=1)
    created_at = Column(String(64), nullable=False)

    requests = relationship("CSRRequest", back_populates="partner")


class CSRRequest(Base):
    __tablename__ = "csr_requests"

    request_id = Column(String(128), primary_key=True)
    company_id = Column(String(128), ForeignKey("csr_partners.company_id"), nullable=False)
    pothole_event_id = Column(String(128), nullable=False)
    bus_id = Column(String(128), nullable=False)
    timestamp = Column(Float, nullable=False)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    depth_cm = Column(Float, nullable=True)
    area_m2 = Column(Float, nullable=True)
    severity = Column(String(64), nullable=True)
    road_name = Column(String(255), nullable=True)
    estimated_repair_cost_inr = Column(Float, nullable=False)
    requested_amount_inr = Column(Float, nullable=False)
    evidence_image = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    status = Column(String(64), nullable=False)
    created_at = Column(String(64), nullable=False)
    updated_at = Column(String(64), nullable=False)

    partner = relationship("CSRPartner", back_populates="requests")

    __table_args__ = (
        Index("idx_csr_requests_pothole", "pothole_event_id"),
        Index("idx_csr_requests_company", "company_id"),
        Index("idx_csr_requests_status", "status"),
    )


class ReferenceAsset(Base):
    __tablename__ = "reference_assets"

    asset_id = Column(String(128), primary_key=True)
    road_segment_id = Column(String(128), nullable=False)
    asset_type = Column(String(64), nullable=False)
    asset_subtype = Column(String(64), nullable=True)
    expected_state = Column(String(64), nullable=False)
    reference_source = Column(String(128), nullable=False)
    source_quality = Column(String(64), default="DEMO")
    geometry_json = Column(Text, nullable=True)
    allowed_openings_json = Column(Text, nullable=True)
    last_verified = Column(String(64), nullable=True)
    created_at = Column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_ref_assets_type", "asset_type"),
    )


class InfrastructureObservation(Base):
    __tablename__ = "infrastructure_observations"

    observation_id = Column(String(128), primary_key=True)
    bus_id = Column(String(128), nullable=False)
    timestamp = Column(Float, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)
    road_segment_id = Column(String(128), nullable=True)
    event_type = Column(String(64), nullable=False)
    track_id = Column(Integer, nullable=True)
    model_confidence = Column(Float, nullable=False)
    visibility_score = Column(Float, nullable=False)
    temporal_consistency = Column(Float, default=1.0)
    parameters_json = Column(Text, nullable=False)
    evidence_refs_json = Column(Text, nullable=False)
    simulation_mode = Column(Integer, default=0)
    created_at = Column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_infra_obs_type", "event_type"),
        Index("idx_infra_obs_bus", "bus_id"),
    )


class InfrastructureEvent(Base):
    __tablename__ = "infrastructure_events"

    event_id = Column(String(128), primary_key=True)
    event_type = Column(String(64), nullable=False)
    road_segment_id = Column(String(128), nullable=True)
    reference_asset_id = Column(String(128), nullable=True)
    status = Column(String(64), nullable=False)
    first_seen = Column(Float, nullable=False)
    last_seen = Column(Float, nullable=False)
    observation_count = Column(Integer, default=1)
    unique_bus_count = Column(Integer, default=1)
    unique_pass_count = Column(Integer, default=1)
    unique_day_count = Column(Integer, default=1)
    visibility_summary_json = Column(Text, nullable=True)
    temporal_consistency = Column(Float, default=1.0)
    spatial_consistency = Column(Float, default=1.0)
    reference_support = Column(String(64), default="UNKNOWN")
    conflict_count = Column(Integer, default=0)
    parameters_json = Column(Text, nullable=False)
    evidence_refs_json = Column(Text, nullable=False)
    simulation_mode = Column(Integer, default=0)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)
    bus_ids_json = Column(Text, nullable=False)
    created_at = Column(String(64), nullable=False)
    updated_at = Column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_infra_events_type", "event_type"),
        Index("idx_infra_events_status", "status"),
        Index("idx_infra_events_seg", "road_segment_id"),
    )


class BusTelemetry(Base):
    __tablename__ = "bus_telemetry"

    telemetry_id = Column(Integer, primary_key=True, autoincrement=True)
    bus_id = Column(String(128), nullable=False)
    timestamp = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    heading = Column(Float, nullable=False)
    road_name = Column(String(255), nullable=False)
    osm_way_id = Column(Integer, nullable=True)
    road_segment_id = Column(String(128), nullable=False)
    distance_along_road_m = Column(Float, nullable=False)
    distance_to_centerline_m = Column(Float, nullable=False)
    speed_kmh = Column(Float, nullable=False)
    simulation_mode = Column(Integer, default=1)
    recorded_at = Column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_bus_telem_bus", "bus_id"),
    )


class EventModel(Base):
    __tablename__ = "events"

    event_id = Column(String(128), primary_key=True)
    bus_id = Column(String(128), nullable=False)
    timestamp = Column(Float, nullable=False)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    route_id = Column(String(128), nullable=False)
    event_type = Column(String(64), nullable=False)
    severity = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    source_modality = Column(String(64), nullable=False)
    software_version = Column(String(64), nullable=False)
    road_segment_id = Column(String(128), nullable=True)
    payload_json = Column(Text, nullable=False)
    created_at = Column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_events_type", "event_type"),
        Index("idx_events_bus", "bus_id"),
        Index("idx_events_time", "timestamp"),
    )


class BusHeartbeat(Base):
    __tablename__ = "bus_heartbeats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bus_id = Column(String(128), nullable=False)
    timestamp = Column(Float, nullable=False)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    speed_kmh = Column(Float, nullable=False)
    system_status = Column(String(64), nullable=False)
    payload_json = Column(Text, nullable=False)
    received_at = Column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_heartbeats_bus", "bus_id"),
    )


class MediaRecord(Base):
    __tablename__ = "media_records"

    media_id = Column(String(128), primary_key=True)
    event_id = Column(String(128), nullable=False)
    bus_id = Column(String(128), nullable=False)
    media_type = Column(String(64), nullable=False)
    mime_type = Column(String(64), nullable=False)
    filename = Column(String(255), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    checksum_sha256 = Column(String(128), nullable=False)
    file_path = Column(Text, nullable=False)
    uploaded_at = Column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_media_event_id", "event_id"),
    )
