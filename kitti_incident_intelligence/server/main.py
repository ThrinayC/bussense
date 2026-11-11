"""
FastAPI Central Validation Server — Phase 10
Provides thin API endpoints for incident events and VRU spatial observations per PRD §49.
"""

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from server.models import IncidentEventSchema, VRUObservationSchema

app = FastAPI(
    title="BusSense Incident & VRU Validation API",
    version="2.0.0",
    description="Standalone validation backend for mobile road safety sensing unit events and spatial layers.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory spatial store for validation
EVENTS_DB: Dict[str, IncidentEventSchema] = {}
VRU_OBS_DB: Dict[str, VRUObservationSchema] = {}


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "BusSense_Validation_Server",
        "total_events": len(EVENTS_DB),
        "total_vru_observations": len(VRU_OBS_DB),
    }


@app.post("/api/events", status_code=status.HTTP_201_CREATED)
def ingest_event(event: IncidentEventSchema):
    EVENTS_DB[event.event_id] = event
    return {"status": "ACK", "event_id": event.event_id, "state": event.state}


@app.get("/api/events", response_model=List[IncidentEventSchema])
def list_events(event_type: Optional[str] = None, limit: int = 50):
    all_events = list(EVENTS_DB.values())
    if event_type:
        all_events = [e for e in all_events if e.event_type == event_type]
    return all_events[:limit]


@app.get("/api/events/{event_id}", response_model=IncidentEventSchema)
def get_event_by_id(event_id: str):
    if event_id not in EVENTS_DB:
        raise HTTPException(status_code=404, detail="Event not found")
    return EVENTS_DB[event_id]


@app.post("/api/vru-observations", status_code=status.HTTP_201_CREATED)
def ingest_vru_observation(obs: VRUObservationSchema):
    VRU_OBS_DB[obs.observation_id] = obs
    return {"status": "ACK", "observation_id": obs.observation_id}


@app.get("/api/vru-areas")
def get_vru_vulnerable_areas(min_observations: int = 2):
    """
    Spatially aggregates VRU observations into vulnerable road segments.
    """
    segments: Dict[str, List[VRUObservationSchema]] = {}
    for obs in VRU_OBS_DB.values():
        seg_id = obs.road_segment_id or "UNASSIGNED"
        if seg_id not in segments:
            segments[seg_id] = []
        segments[seg_id].append(obs)

    vulnerable_areas = []
    for seg_id, obs_list in segments.items():
        n = len(obs_list)
        if n < min_observations:
            continue

        crossing_count = sum(1 for o in obs_list if o.crossing)
        in_path_count = sum(1 for o in obs_list if o.in_path)
        near_miss_count = sum(1 for o in obs_list if o.near_miss)

        vai_score = round(float((crossing_count * 1.0) + (in_path_count * 2.0) + (near_miss_count * 5.0)), 2)

        vulnerable_areas.append({
            "segment_id": seg_id,
            "observation_count": n,
            "crossing_count": crossing_count,
            "in_path_count": in_path_count,
            "near_miss_count": near_miss_count,
            "vai_score": vai_score,
            "center_lat": obs_list[0].gps.lat,
            "center_lon": obs_list[0].gps.lon,
        })

    return {"total_vulnerable_segments": len(vulnerable_areas), "segments": vulnerable_areas}
