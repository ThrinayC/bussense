# BusSense FastAPI Server Contract & Lightweight Ingestion Specification

## 1. Architectural Philosophy: Edge Autonomy vs Central Aggregation

The BusSense platform is built upon a strict separation between **edge perception computing** and **central server ingestion**:

```
EDGE DEVICE (Mobile Bus Unit):
==============================
LEFT + RIGHT RGB Stereo Cameras
             ↓
Local Perception Pipelines (Intact)
  • SGBM Real Stereo Disparity
  • 3D Obstacle Tracking (Kalman)
  • Road Plane RANSAC Metrology
  • YOLO Pothole/Crack Segmentation
  • EasyOCR-Ensemble Multi-Variant ANPR
             ↓
Local Event Filter & Telemetry Gatekeeper
  • Discards normal cruising vehicles (>98% of frames)
  • Keeps raw disparity, depths, and Kalman states LOCAL
  • Enforces 30s rolling aggregation for traffic dynamics
  • Enforces periodic 15s lightweight bus heartbeats
  • Uploads ONLY verified defect/incident packages
             ↓
       Upload Queue
             ↓
      FastAPI Backend
```

```
CENTRAL SERVER:
===============
FastAPI Ingestion Endpoints
             ↓
Strict Pydantic Schema Validation (HTTP 422 on invalid)
             ↓
Idempotency Resolution (Zero duplicate events)
             ↓
SQLite / File Persistence (`outputs/server_data/bussense_events.db`)
             ↓
GIS Layers / Incident History / Dashboard API
```

### Prohibited Telemetry (Kept Strictly on Edge)
- Continuous raw vehicle detection streams (`vehicle_17_distance_every_frame`)
- Per-frame 3D bounding boxes and raw pixel coordinates
- Per-frame metric distances and TTC samples
- Raw stereo disparity and depth maps
- Continuous uncut camera video streams

---

## 2. Common Event Envelope Specification

Every safety, defect, or telemetry event inherits the `CommonEventEnvelope` Pydantic model:

| Field | Type | Validation Rule | Description |
| :--- | :--- | :--- | :--- |
| `event_id` | `str` | Length 3–128 | Globally unique identifier (Primary Key) |
| `bus_id` | `str` | Length 2–64 | Mobile unit identifier (e.g. `BUS-042`) |
| `timestamp` | `float` | $> 0.0$ | POSIX timestamp or calibrated encounter time (s) |
| `gps_lat` | `float` | $[-90.0, +90.0]$ | GPS Latitude in decimal degrees |
| `gps_lon` | `float` | $[-180.0, +180.0]$ | GPS Longitude in decimal degrees |
| `route_id` | `str` | Length 1–64 | Active corridor/transit route identifier |
| `event_type` | `str` | Enumerated | Specific event classification |
| `severity` | `str` | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | Operational severity index |
| `confidence` | `float` | $[0.0, 1.0]$ | Statistical detection/ensemble confidence |
| `source_modality` | `str` | `stereo`, `monocular`, `sensor_fusion` | Sensor perception modality |
| `software_version`| `str` | Length 1–32 | Edge firmware/release version (`bussense-2.4.0`) |
| `heading` | `float` | $[0.0, 360.0]$ (optional) | Compass heading angle |
| `road_segment_id` | `str` | Length $\le 64$ (optional) | GIS street/corridor segment ID |
| `sequence_id` | `str` | Length $\le 64$ (optional) | Trip or dataset sequence ID |

---

## 3. Endpoints & Payload Contracts

### A. Bus Heartbeat (`POST /api/heartbeat`)
- **Interval**: Configurable 10–30s (Default: 15s).
- **Payload Size**: $\approx 220$ bytes.
- **Fields**: `bus_id`, `timestamp`, `gps_lat`, `gps_lon`, `speed_kmh`, `heading`, `system_status`, `camera_status`, `gps_status`, `storage_status`, `network_status`, `software_version`.
- **Response**: `HTTP 201 Created` with `{"status": "ACK", "heartbeat_id": <int>}`.

### B. Road Defect Events (`POST /api/events/road`)
- **Potholes**:
  - Requires positive metric geometry: `depth_cm > 0.0`, `area_m2 > 0.0`.
  - Invalid geometry detections stay **local and silent**.
  - Attached media: **ONE representative frame** (`image`).
- **Cracks**:
  - Requires: `length_m > 0.0`, `branches >= 0`, `road_condition`.
  - Attached media: **ONE representative frame**.
- **Response**: `HTTP 201 Created` on new event, `HTTP 200 OK` on duplicate submission (Idempotency).

### C. Critical Incident Events (`POST /api/events/incident`)
- **Event Types**: `RAPID_CLOSING`, `DANGEROUS_FOLLOWING`, `HARD_BRAKING`, `CUT_IN_CANDIDATE`, `ABRUPT_LATERAL_MANEUVER`, `WEAVING_CANDIDATE`, `COLLISION_CANDIDATE`, `EMERGENCY_EVASIVE_MANEUVER`, `HIT_AND_RUN_CANDIDATE`.
- **Kinematic Fields**: `primary_track_id`, `object_class`, `range_m`, `relative_speed_mps`, `ttc_s`.
- **Hit-and-Run Specifics**:
  - `source_modality = "monocular"` (strictly required).
  - `metric_depth_valid = false` (zero fake stereo claims).
  - `suspect_track_id`, `victim_track_id`, `departure_detected`, `registration_number`, `ocr_confidence`.
- **Attached Media**: Single event frame + **short $\approx$ 8-second clip** (`3s pre-event + 2s event + 3s post-event`).

### D. VRU Encounters (`POST /api/events/vru`)
- **Event Types**: `PEDESTRIAN_IN_PATH`, `CYCLIST_IN_PATH`, `VRU_NEAR_MISS`, `SUDDEN_VRU_EMERGENCE`.
- **Fields**: `vru_type`, `track_id`, `range_m`, `ttc_s`, `image`, `event_clip`.
- **Policy**: Normal sidewalk pedestrians are filtered out locally.

### E. Verified ANPR Events (`POST /api/events/anpr`)
- **Strict Verification Contract**:
  - Requires: `registration_number` (min 4 characters), `ocr_confidence >= 0.40`, `plate_confidence >= 0.40`, `plate_pixel_width >= 20.0`.
  - **Rejection Policy**: Submissions with `"UNREADABLE"`, `"UNKNOWN"`, `"NO PLATE"`, or empty strings are **strictly rejected with HTTP 422**.
  - No continuous streaming: exactly 1 upload per verified vehicle track.

### F. Traffic Aggregates (`POST /api/events/traffic`)
- **Rolling Window**: 30 seconds (or on significant jam-risk state transition).
- **Fields**: `flow_rate_vpm`, `density_veh_per_km`, `mean_speed_kmh`, `median_speed_kmh`, `occupancy`, `mean_headway_s`, `queue_length_m`, `traffic_state` (`FREE_FLOW`, `BUILDING`, `CONGESTED`, `JAM_RISK`), `jam_risk` ($[0.0, 1.0]$).

### G. Controlled Media Upload (`POST /api/media/upload`)
- **Storage Structure**: Deterministic path: `outputs/server_media/events/<bus_id>/<date>/<event_id>/<filename>`.
- **Size Limit**: 25 MB per file.
- **Allowed MIME**: `image/jpeg`, `image/png`, `image/webp`, `video/mp4`, `video/webm`.
- **Security**: Strict path sanitization and SHA-256 checksum verification.

---

## 4. Idempotency & Offline Resilience

1. **Idempotency Contract**:
   - The server maintains a unique index on `event_id`.
   - When an event is uploaded multiple times due to cellular retry, the server detects the duplicate, skips re-insertion, and returns `HTTP 200 OK` with `"status": "IDEMPOTENT_ACK"` and `"is_duplicate": true`.
2. **Offline Queue Contract**:
   - The edge client [`bussense_edge/offline_queue.py`](file:///C:/Users/Dharun%20Kumar%20S/.gemini/antigravity/scratch/bussense_edge/offline_queue.py) buffers pending events to local disk (`outputs/edge_queue/`) during network outages.
   - Upon connection restoration, pending events are dispatched and cleared only after receiving confirmed HTTP ACK.
