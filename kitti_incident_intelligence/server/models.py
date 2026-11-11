"""
Pydantic Schemas for Central Event & Spatial VRU Layers
Strictly adheres to PRD §47 and §48 contracts.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class GpsCoordinates(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None


class EgoTelemetry(BaseModel):
    speed_mps: Optional[float] = None
    yaw_rate: Optional[float] = None


class KinematicMetrics(BaseModel):
    distance_m: Optional[float] = None
    relative_speed_mps: Optional[float] = None
    closing_rate_mps: Optional[float] = None
    time_headway_s: Optional[float] = None
    ttc_kinematic_s: Optional[float] = None
    ttc_looming_s: Optional[float] = None


class EvidencePackage(BaseModel):
    pre_event: List[str] = []
    event: List[str] = []
    post_event: List[str] = []
    clip: Optional[str] = None
    vehicle_crop: Optional[str] = None
    plate_crop: Optional[str] = None


class RegistrationData(BaseModel):
    number: Optional[str] = None
    confidence: Optional[float] = None


class IncidentEventSchema(BaseModel):
    event_id: str
    event_type: str
    state: str = "UNVERIFIED"
    timestamp: float
    frame_id: int
    gps: GpsCoordinates
    ego: EgoTelemetry
    vehicles: List[Dict[str, Any]] = []
    vru: Dict[str, Any] = {"track_id": None, "class": None}
    metrics: KinematicMetrics
    uncertainty: Dict[str, Any] = {}
    registration: RegistrationData = RegistrationData()
    evidence: EvidencePackage = EvidencePackage()


class VRUObservationSchema(BaseModel):
    observation_id: str
    timestamp: float
    gps: GpsCoordinates
    road_segment_id: Optional[str] = None
    vru_class: str = "person"
    crossing: bool = False
    in_path: bool = False
    near_miss: bool = False
    small_stature_candidate: bool = False
    distance_m: float
    ttc_s: Optional[float] = None
