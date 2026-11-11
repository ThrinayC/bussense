"""
BusSense FastAPI Backend — Schemas & Data Contract
==================================================
Defines strongly-validated Pydantic models for the Common Event Envelope,
Road Defects, Critical Incidents, VRU Events, Verified ANPR, Traffic Aggregates,
Vulnerable Areas, Heartbeats, and Media Metadata.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# 1. COMMON EVENT ENVELOPE
# =============================================================================

class CommonEventEnvelope(BaseModel):
    """
    Standard common envelope for all BusSense edge-to-server events.
    Strictly validated per Section 3 of the Server Contract.
    """
    event_id: str = Field(..., min_length=3, max_length=128, description="Globally unique event identifier")
    bus_id: str = Field(..., min_length=2, max_length=64, description="Unique bus or edge device identifier")
    timestamp: float = Field(..., gt=0.0, description="POSIX epoch timestamp or calibrated run time (seconds)")
    gps_lat: Optional[float] = Field(None, ge=-90.0, le=90.0, description="GPS Latitude in decimal degrees [-90, +90]")
    gps_lon: Optional[float] = Field(None, ge=-180.0, le=180.0, description="GPS Longitude in decimal degrees [-180, +180]")
    route_id: str = Field(..., min_length=1, max_length=64, description="Transit route or patrol corridor identifier")
    event_type: str = Field(..., min_length=2, max_length=64, description="Specific event classification")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(..., description="Operational severity classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model or ensemble detection confidence [0.0, 1.0]")
    source_modality: Literal["stereo", "monocular", "sensor_fusion"] = Field(..., description="Perception sensor modality")
    software_version: str = Field("bussense-2.4.0", min_length=1, max_length=32, description="Edge software release version")

    # Optional spatial/sequence metadata
    heading: Optional[float] = Field(None, ge=0.0, le=360.0, description="Vehicle heading angle in degrees [0, 360]")
    road_segment_id: Optional[str] = Field(None, max_length=64, description="GIS road segment identifier")
    sequence_id: Optional[str] = Field(None, max_length=64, description="Dataset or trip sequence identifier")
    demo_route_location: Optional[Dict[str, Any]] = Field(None, description="Presentation demo route coordinates {lat, lon, label}")
    source_gps: Optional[Dict[str, Any]] = Field(None, description="Original source dataset benchmark GPS metadata")


# =============================================================================
# 2. BUS HEARTBEAT
# =============================================================================

class BusHeartbeatSchema(BaseModel):
    """
    Lightweight periodic telemetry for mobile edge bus units (Section 13).
    Transmitted every 10-30 seconds with minimal network payload.
    """
    bus_id: str = Field(..., min_length=2, max_length=64)
    timestamp: float = Field(..., gt=0.0)
    gps_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    gps_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    speed_kmh: float = Field(..., ge=0.0, le=200.0)
    heading: Optional[float] = Field(None, ge=0.0, le=360.0)

    system_status: Literal["OK", "WARNING", "DEGRADED", "ERROR"] = "OK"
    camera_status: Literal["OK", "OCCLUDED", "DEGRADED", "ERROR"] = "OK"
    gps_status: Literal["OK", "LOST", "DEGRADED"] = "OK"
    storage_status: Literal["OK", "NEAR_FULL", "FULL"] = "OK"
    network_status: Literal["ONLINE", "OFFLINE_BUFFERING", "POOR"] = "ONLINE"

    software_version: str = "bussense-2.4.0"


# =============================================================================
# 3. ROAD / DEFECT EVENTS (POTHOLE & CRACK)
# =============================================================================

class RoadEventSchema(CommonEventEnvelope):
    """
    Validated road surface defects (Pothole & Crack) per Sections 4 & 5.
    Uploaded ONLY when valid geometry exists.
    """
    # Allowed road event types
    event_type: Literal["POTHOLE", "CRACK"] = "POTHOLE"
    road_condition: Literal["LOW", "NORMAL", "MEDIUM", "HIGH", "CRITICAL"] = "NORMAL"

    # Pothole fields
    pothole_id: Optional[int] = Field(None, ge=0)
    depth_cm: Optional[float] = Field(None, gt=0.0, le=50.0, description="Metric cavity depth (cm)")
    area_m2: Optional[float] = Field(None, gt=0.0, le=25.0, description="Metric surface area (m²)")

    # Crack fields
    crack_id: Optional[int] = Field(None, ge=0)
    length_m: Optional[float] = Field(None, gt=0.0, le=100.0, description="Linear crack length (m)")
    branches: Optional[int] = Field(None, ge=0, le=500, description="Crack network branch count")

    # Image & clip metadata
    image: Optional[str] = Field(None, description="Path or media reference ID of single representative frame")
    event_clip: Optional[str] = Field(None, description="Short evidence clip reference (~8s)")

    @model_validator(mode="after")
    def validate_defect_fields(self):
        if self.event_type == "POTHOLE":
            if self.depth_cm is None or self.area_m2 is None:
                raise ValueError("POTHOLE event requires valid positive depth_cm and area_m2")
        elif self.event_type == "CRACK":
            if self.length_m is None:
                raise ValueError("CRACK event requires valid positive length_m")
        return self


# =============================================================================
# 4. INCIDENT EVENTS (TRAFFIC & HIT-AND-RUN)
# =============================================================================

class IncidentEventSchema(CommonEventEnvelope):
    """
    Critical kinematic and pre-collision incident events per Section 6 & 9.
    """
    event_type: Literal[
        "RAPID_CLOSING",
        "DANGEROUS_FOLLOWING",
        "HARD_BRAKING",
        "CUT_IN_CANDIDATE",
        "ABRUPT_LATERAL_MANEUVER",
        "WEAVING_CANDIDATE",
        "COLLISION_CANDIDATE",
        "EMERGENCY_EVASIVE_MANEUVER",
        "HIT_AND_RUN_CANDIDATE"
    ]

    primary_track_id: Optional[int] = Field(None, ge=0)
    object_class: str = Field("vehicle", max_length=32)

    # Valid metric kinematics (strictly non-negative when valid)
    range_m: Optional[float] = Field(None, ge=0.0, le=150.0)
    relative_speed_mps: Optional[float] = Field(None, ge=-100.0, le=100.0)
    ttc_s: Optional[float] = Field(None, ge=0.0, le=30.0)

    # Hit-and-Run specific fields
    metric_depth_valid: bool = Field(True, description="False for monocular incident footage")
    suspect_track_id: Optional[int] = Field(None, ge=0)
    victim_track_id: Optional[int] = Field(None, ge=0)
    event_stage: Optional[str] = Field(None, max_length=64)
    departure_detected: Optional[bool] = None
    registration_number: Optional[str] = Field(None, max_length=16)
    ocr_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Media attachments
    image: Optional[str] = Field(None, description="Representative event frame reference")
    event_clip: Optional[str] = Field(None, description="Short evidence clip reference (~8s)")

    @model_validator(mode="after")
    def validate_incident_modality(self):
        if self.event_type == "HIT_AND_RUN_CANDIDATE":
            if self.source_modality != "monocular":
                raise ValueError("HIT_AND_RUN_CANDIDATE must have source_modality='monocular'")
            if self.metric_depth_valid:
                raise ValueError("HIT_AND_RUN_CANDIDATE must declare metric_depth_valid=False")
        return self


# =============================================================================
# 5. VRU EVENTS
# =============================================================================

class VRUEventSchema(CommonEventEnvelope):
    """
    Vulnerable Road User encounter and near-miss events per Section 7.
    """
    event_type: Literal[
        "PEDESTRIAN_IN_PATH",
        "CYCLIST_IN_PATH",
        "VRU_NEAR_MISS",
        "SUDDEN_VRU_EMERGENCE"
    ]
    vru_type: Literal["pedestrian", "cyclist", "rider", "vru"] = "pedestrian"
    track_id: int = Field(..., ge=0)

    range_m: Optional[float] = Field(None, ge=0.0, le=100.0)
    ttc_s: Optional[float] = Field(None, ge=0.0, le=20.0)

    image: Optional[str] = Field(None, description="Representative frame")
    event_clip: Optional[str] = Field(None, description="Short clip for near-miss/emergency")


# =============================================================================
# 6. ANPR / LICENSE PLATE EVENTS
# =============================================================================

class ANPREventSchema(CommonEventEnvelope):
    """
    Verified Vehicle Registration Events per Section 8.
    Strictly requires verified plate reading. Rejects 'UNREADABLE' or failed OCR.
    """
    event_type: Literal["ANPR_VERIFIED"] = "ANPR_VERIFIED"
    track_id: int = Field(..., ge=0)
    vehicle_class: str = Field("car", max_length=32)

    registration_number: str = Field(..., min_length=4, max_length=16, description="Verified license plate string")
    ocr_confidence: float = Field(..., ge=0.40, le=1.0, description="OCR model recognition confidence")
    plate_confidence: float = Field(..., ge=0.40, le=1.0, description="Plate candidate detection confidence")

    plate_pixel_width: float = Field(..., ge=20.0, description="Candidate plate pixel width")
    best_frame: Optional[str] = Field(None, description="Crop or image reference of best plate frame")
    image: Optional[str] = Field(None, description="Full vehicle frame reference")
    event_clip: Optional[str] = Field(None, description="Playable short evidence video clip (~3-6s)")

    ocr_engine: str = "EasyOCR-Ensemble"
    ocr_consensus_count: int = Field(1, ge=1)

    @field_validator("registration_number")
    @classmethod
    def validate_registration(cls, v: str) -> str:
        cleaned = v.strip().upper()
        forbidden = ["UNREADABLE", "UNKNOWN", "NO PLATE", "N/A", "OCR FAILED", "NULL", "FAILED"]
        if any(f in cleaned for f in forbidden) or len(cleaned) < 4:
            raise ValueError(f"Registration number '{v}' is invalid or unreadable. Silent local retention required.")
        return cleaned


# =============================================================================
# 7. TRAFFIC AGGREGATE EVENTS
# =============================================================================

class TrafficEventSchema(CommonEventEnvelope):
    """
    Spatially and temporally aggregated traffic dynamics per Section 10 & 11.
    Transmitted on 30s rolling windows or major state transitions.
    """
    event_type: Literal["TRAFFIC_AGGREGATE"] = "TRAFFIC_AGGREGATE"

    flow_rate_vpm: float = Field(..., ge=0.0, le=500.0, description="Virtual gate flow rate (veh/min)")
    density_veh_per_km: float = Field(..., ge=0.0, le=1000.0, description="Spatial corridor density (veh/km)")
    mean_speed_kmh: float = Field(..., ge=0.0, le=200.0, description="Mean vehicle stream speed (km/h)")
    median_speed_kmh: float = Field(..., ge=0.0, le=200.0, description="Median vehicle stream speed (km/h)")

    occupancy: float = Field(..., ge=0.0, le=1.0, description="Corridor footprint occupancy [0.0, 1.0]")
    mean_headway_s: Optional[float] = Field(None, ge=0.0, le=120.0, description="Mean inter-vehicle time headway (s)")
    queue_length_m: Optional[float] = Field(None, ge=0.0, le=500.0, description="Estimated queue length (m)")

    traffic_state: Literal["FREE_FLOW", "BUILDING", "CONGESTED", "JAM_RISK"] = Field(..., description="Operational state")
    jam_risk: float = Field(..., ge=0.0, le=1.0, description="Predictive jam risk score [0.0, 1.0]")

    measurement_window_s: float = Field(30.0, ge=5.0, le=300.0, description="Aggregation time window (s)")

    # Image evidence for CONGESTED and JAM_RISK
    image: Optional[str] = Field(None, description="Representative congestion/jam-risk evidence frame reference")
    evidence_image_url: Optional[str] = Field(None, description="Direct URL to congestion evidence image")


# =============================================================================
# 8. VULNERABLE AREA AGGREGATE EVENTS
# =============================================================================

class VulnerableAreaSchema(CommonEventEnvelope):
    """
    Spatial summary of VRU exposure and conflict density per Section 12.
    """
    event_type: Literal["VULNERABLE_AREA_AGGREGATE"] = "VULNERABLE_AREA_AGGREGATE"
    segment_id: str = Field(..., min_length=1, max_length=64)

    pedestrian_count: int = Field(..., ge=0)
    cyclist_count: int = Field(..., ge=0)
    crossing_count: int = Field(..., ge=0)
    conflict_count: int = Field(..., ge=0)
    near_miss_count: int = Field(..., ge=0)

    bus_pass_count: int = Field(1, ge=1)
    vulnerability_index: float = Field(..., ge=0.0, le=100.0)


# =============================================================================
# 9. MEDIA METADATA
# =============================================================================

class MediaMetadata(BaseModel):
    """
    Media file attachment metadata per Section 14.
    """
    media_id: str
    event_id: str
    bus_id: str
    media_type: Literal["image", "video_clip"]
    mime_type: str
    filename: str
    size_bytes: int
    checksum_sha256: str
    file_path: str


# =============================================================================
# 10. CSR FUNDING & REPAIR WORKFLOW SCHEMAS
# =============================================================================

class CSRPartnerSchema(BaseModel):
    """
    Corporate Social Responsibility sponsor profile.
    """
    company_id: str = Field(..., description="Unique company identifier (e.g. TVS_MOBILITY_CSR)")
    company_name: str = Field(..., description="Company or Foundation Name")
    logo: Optional[str] = Field(None, description="Emoji or logo URL")
    csr_budget_inr: float = Field(..., gt=0.0, description="Total annual CSR allocation in INR")
    available_budget_inr: float = Field(..., ge=0.0, description="Remaining uncommitted road safety budget in INR")
    focus_areas: List[str] = Field(..., description="Focus areas (e.g. ['Road Safety', 'Urban Transit'])")
    geography: str = Field(..., description="Supported geographic corridors (e.g. Tiruchirappalli & Tamil Nadu)")
    contact_name: str = Field(..., description="CSR officer or coordinator name")
    contact_email: str = Field(..., description="Contact email address")
    contact_phone: str = Field(..., description="Contact phone number")
    website: Optional[str] = Field(None, description="Official company sustainability/CSR URL")
    demo_flag: bool = Field(True, description="Explicit indicator for demo/prototype sponsor entity")


class CSRRequestCreateSchema(BaseModel):
    """
    Payload for creating a road repair sponsorship request tied to a real pothole event.
    """
    company_id: str = Field(..., description="ID of the target CSR partner")
    pothole_event_id: str = Field(..., description="Authoritative BusSense pothole event ID (e.g. EVT-POT-006)")
    bus_id: str = Field("BUS-01", description="Sensing bus unit")
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    depth_cm: Optional[float] = None
    area_m2: Optional[float] = None
    severity: Optional[str] = None
    road_name: Optional[str] = None
    estimated_repair_cost_inr: float = Field(..., gt=0.0, description="Calculated engineering estimate in INR")
    requested_amount_inr: float = Field(..., gt=0.0, description="Requested sponsorship amount in INR")
    evidence_image: Optional[str] = Field(None, description="URL or media ID of the full frame evidence")
    notes: Optional[str] = Field("", description="Operator justification or special priority remarks")


class CSRRequestSchema(CSRRequestCreateSchema):
    """
    Stored CSR sponsorship request record with status lifecycle.
    """
    request_id: str = Field(..., description="Unique CSR request ID (e.g. CSR-REQ-2026-003)")
    timestamp: float = Field(..., description="Creation epoch timestamp")
    status: Literal["REQUESTED", "UNDER REVIEW", "APPROVED", "REJECTED", "FUNDED", "COMPLETED"] = "REQUESTED"
    company_name: Optional[str] = None
    created_at: str
    updated_at: str


# =============================================================================
# 12. INFRASTRUCTURE & WATERLOGGING SCHEMAS (Sections 14, 15, 18, 19, 23.6)
# =============================================================================

class AllowedOpening(BaseModel):
    """Authorized opening in a physical median (U-Turn, intersection, access opening)."""
    start_m: float = Field(..., ge=0.0, description="Start distance in local road coordinates")
    end_m: float = Field(..., ge=0.0, description="End distance in local road coordinates")
    type: Literal["U_TURN", "INTERSECTION", "AUTHORIZED_OPENING", "ACCESS"] = "U_TURN"


class ReferenceAsset(BaseModel):
    """
    Known or expected infrastructure asset inventory record (Section 15).
    """
    asset_id: str = Field(..., description="Unique asset ID (e.g. REF-SIGN-001)")
    road_segment_id: str = Field(..., description="Corridor road segment ID")
    asset_type: Literal["TRAFFIC_SIGN", "ZEBRA_CROSSING", "ROAD_DIVIDER", "OTHER_INFRASTRUCTURE"]
    asset_subtype: Optional[str] = None
    expected_state: Literal["YES", "NO", "UNKNOWN"] = Field("YES", description="Expected presence status")
    reference_source: str = Field(..., description="Source hierarchy (OSM, MAPILLARY, OFFICIAL_INVENTORY, MANUAL_DEMO_REFERENCE)")
    source_quality: Literal["HIGH", "MEDIUM", "LOW", "DEMO"] = "DEMO"
    geometry: Optional[Dict[str, Any]] = Field(None, description="Spatial coordinates or local road stationing")
    allowed_openings: List[AllowedOpening] = Field(default_factory=list, description="Authorized gaps/openings for dividers")
    last_verified: Optional[str] = None


class BusSenseObservation(BaseModel):
    """
    Unified Observation Schema (Section 18).
    Captures raw or tracked AI perception prior to multi-pass fusion.
    """
    observation_id: str = Field(..., description="Globally unique observation ID")
    bus_id: str = Field(..., description="Sensing bus ID")
    timestamp: float = Field(..., gt=0.0, description="Observation timestamp")
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    heading: Optional[float] = Field(None, ge=0.0, le=360.0)
    road_segment_id: Optional[str] = None
    event_type: str = Field(..., description="Perception class (WATERLOGGING, TRAFFIC_SIGN, ZEBRA_CROSSING, ROAD_DIVIDER, etc.)")
    track_id: Optional[int] = None

    model_confidence: float = Field(..., ge=0.0, le=1.0)
    visibility_score: float = Field(..., ge=0.0, le=1.0)
    temporal_consistency: float = Field(1.0, ge=0.0, le=1.0)

    # Physical parameters: use None if unavailable, NEVER fake zero
    parameters: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    simulation_mode: bool = Field(False, description="Explicit flag for synthetic/simulated observations")


class InfrastructureEvent(BaseModel):
    """
    Unified Infrastructure Event Schema (Section 19).
    Stateful consolidated event after road-segment association and multi-pass fusion.
    """
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Consolidated event classification")
    road_segment_id: Optional[str] = None
    reference_asset_id: Optional[str] = None

    status: Literal[
        "OBSERVED",
        "TRACKED",
        "PERSISTENT",
        "CANDIDATE",
        "CORROBORATED",
        "VERIFIED",
        "INSUFFICIENT_DATA",
        "CONFLICTING_EVIDENCE",
        "NO_MISSING_CLAIM"
    ] = "OBSERVED"

    first_seen: float = Field(..., gt=0.0)
    last_seen: float = Field(..., gt=0.0)

    observation_count: int = Field(1, ge=1)
    unique_bus_count: int = Field(1, ge=1)
    unique_pass_count: int = Field(1, ge=1)
    unique_day_count: int = Field(1, ge=1)

    visibility_summary: Dict[str, Any] = Field(default_factory=dict)
    temporal_consistency: float = Field(1.0, ge=0.0, le=1.0)
    spatial_consistency: float = Field(1.0, ge=0.0, le=1.0)

    reference_support: Literal["YES", "NO", "UNKNOWN"] = "UNKNOWN"
    conflict_count: int = Field(0, ge=0)

    parameters: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    simulation_mode: bool = Field(False, description="True if generated from synthetic/demo stream")

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    heading: Optional[float] = None
    bus_ids: List[str] = Field(default_factory=list)


class BusTelemetrySchema(BaseModel):
    """
    Continuous bus road telemetry strictly constrained to OSM road geometry (Section 23.6).
    """
    bus_id: str = Field(..., description="Bus identifier")
    timestamp: float = Field(..., gt=0.0)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    heading: float = Field(..., ge=0.0, le=360.0)
    road_name: str = Field(..., description="Validated OSM road name")
    osm_way_id: Optional[int] = None
    road_segment_id: str = Field(..., description="Road segment identifier")
    distance_along_road_m: float = Field(..., ge=0.0, description="Distance along road polyline")
    distance_to_centerline_m: float = Field(..., ge=0.0, le=10.0, description="Validated distance to centerline (must be <=10m)")
    speed_kmh: float = Field(25.0, ge=0.0, le=120.0)
    simulation_mode: bool = Field(True, description="Always true for simulated Trichy replay")


