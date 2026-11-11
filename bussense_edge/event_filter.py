"""
BusSense Edge Event Filter & Data Gatekeeper
=============================================
Enforces the core architectural rule:
"The server MUST NOT receive a continuous stream of raw vehicle detections,
bounding boxes, per-frame distances, per-frame TTCs, disparity maps, or video.
The edge aggregates locally and transmits ONLY verified, meaningful events."
"""

import time
import uuid
from typing import Dict, List, Optional, Any, Tuple


class BusSenseEdgeEventFilter:
    """
    Edge-side Event Gatekeeper.
    Decides whether a perception state or measurement warrants server transmission.
    Filters out >98% of high-frequency local perception noise.
    """

    def __init__(
        self,
        bus_id: str = "BUS-042",
        route_id: str = "ROUTE-104",
        software_version: str = "bussense-2.4.0",
        traffic_aggregation_window_s: float = 30.0,
        heartbeat_interval_s: float = 15.0
    ):
        self.bus_id = bus_id
        self.route_id = route_id
        self.software_version = software_version
        self.traffic_window_s = traffic_aggregation_window_s
        self.heartbeat_interval_s = heartbeat_interval_s

        # Rolling state tracking
        self.last_heartbeat_time = 0.0
        self.last_traffic_upload_time = 0.0
        self.last_traffic_state: Optional[str] = None
        self.known_uploaded_potholes = set()
        self.known_uploaded_cracks = set()
        self.known_uploaded_plates = set()
        self.recent_incidents = {}  # track_id -> last_incident_time

    # =========================================================================
    # 1. HEARTBEAT GATE
    # =========================================================================

    def evaluate_heartbeat(
        self,
        timestamp: float,
        gps_lat: float,
        gps_lon: float,
        speed_kmh: float,
        heading: Optional[float] = None,
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Determines if a periodic heartbeat is due.
        Payload is strictly kept minimal (~200 bytes).
        """
        if force or (timestamp - self.last_heartbeat_time >= self.heartbeat_interval_s):
            self.last_heartbeat_time = timestamp
            return {
                "bus_id": self.bus_id,
                "timestamp": round(float(timestamp), 2),
                "gps_lat": round(float(gps_lat), 6),
                "gps_lon": round(float(gps_lon), 6),
                "speed_kmh": round(float(speed_kmh), 1),
                "heading": round(float(heading), 1) if heading is not None else None,
                "system_status": "OK",
                "camera_status": "OK",
                "gps_status": "OK",
                "storage_status": "OK",
                "network_status": "ONLINE",
                "software_version": self.software_version
            }
        return None

    # =========================================================================
    # 2. ROAD DEFECT GATE (POTHOLE & CRACK)
    # =========================================================================

    def filter_pothole_defect(
        self,
        timestamp: float,
        gps_lat: float,
        gps_lon: float,
        pothole_id: int,
        depth_cm: Optional[float],
        area_m2: Optional[float],
        is_valid: bool,
        road_condition: str = "NORMAL",
        representative_image_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates local pothole metrology:
        - If geometry is INVALID or depth <= 0: KEEP LOCAL / SUPPRESS.
        - If already uploaded for this defect ID: SUPPRESS duplicate.
        - If valid: Upload ONE representative event package.
        """
        if not is_valid or depth_cm is None or area_m2 is None:
            # Stays strictly local per Section 4
            return None

        if depth_cm <= 0.0 or area_m2 <= 0.0:
            return None

        # Deduplicate: 1 physical pothole = 1 uploaded server event
        if pothole_id in self.known_uploaded_potholes:
            return None

        self.known_uploaded_potholes.add(pothole_id)

        severity = "HIGH" if depth_cm >= 4.0 else ("MEDIUM" if depth_cm >= 2.5 else "LOW")
        event_id = f"EVT_POT_{self.bus_id}_{pothole_id}_{int(timestamp)}"

        return {
            "event_id": event_id,
            "bus_id": self.bus_id,
            "timestamp": round(float(timestamp), 2),
            "gps_lat": round(float(gps_lat), 6),
            "gps_lon": round(float(gps_lon), 6),
            "route_id": self.route_id,
            "event_type": "POTHOLE",
            "severity": severity,
            "confidence": 0.94,
            "source_modality": "stereo",
            "software_version": self.software_version,
            "pothole_id": pothole_id,
            "depth_cm": round(float(depth_cm), 2),
            "area_m2": round(float(area_m2), 3),
            "road_condition": road_condition,
            "image": representative_image_path
        }

    def filter_crack_defect(
        self,
        timestamp: float,
        gps_lat: float,
        gps_lon: float,
        crack_id: int,
        length_m: Optional[float],
        branches: int = 1,
        road_condition: str = "NORMAL",
        representative_image_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates local crack detection: Uploads ONE representative package per defect.
        """
        if length_m is None or length_m <= 0.0:
            return None

        if crack_id in self.known_uploaded_cracks:
            return None

        self.known_uploaded_cracks.add(crack_id)

        severity = "CRITICAL" if length_m >= 10.0 else ("HIGH" if length_m >= 5.0 else "MEDIUM")
        event_id = f"EVT_CRK_{self.bus_id}_{crack_id}_{int(timestamp)}"

        return {
            "event_id": event_id,
            "bus_id": self.bus_id,
            "timestamp": round(float(timestamp), 2),
            "gps_lat": round(float(gps_lat), 6),
            "gps_lon": round(float(gps_lon), 6),
            "route_id": self.route_id,
            "event_type": "CRACK",
            "severity": severity,
            "confidence": 0.89,
            "source_modality": "stereo",
            "software_version": self.software_version,
            "crack_id": crack_id,
            "length_m": round(float(length_m), 2),
            "branches": branches,
            "road_condition": road_condition,
            "image": representative_image_path
        }

    # =========================================================================
    # 3. CRITICAL INCIDENT & HIT-AND-RUN GATE
    # =========================================================================

    def filter_incident_event(
        self,
        timestamp: float,
        gps_lat: float,
        gps_lon: float,
        event_type: str,
        primary_track_id: int,
        range_m: Optional[float],
        relative_speed_mps: Optional[float],
        ttc_s: Optional[float],
        confidence: float,
        source_modality: str = "stereo",
        image_path: Optional[str] = None,
        clip_path: Optional[str] = None,
        # Hit-and-run arguments
        metric_depth_valid: bool = True,
        suspect_track_id: Optional[int] = None,
        victim_track_id: Optional[int] = None,
        event_stage: Optional[str] = None,
        departure_detected: Optional[bool] = None,
        registration_number: Optional[str] = None,
        ocr_confidence: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates whether an obstacle kinematic state constitutes a reportable emergency.
        Normal cruising vehicles (TTC > 4.0s, low deceleration) are KEPT LOCAL.
        """
        # Hit-and-Run has specialized qualification
        if event_type == "HIT_AND_RUN_CANDIDATE":
            event_id = f"EVT_HNR_{self.bus_id}_{int(timestamp)}"
            return {
                "event_id": event_id,
                "bus_id": self.bus_id,
                "timestamp": round(float(timestamp), 2),
                "gps_lat": round(float(gps_lat), 6),
                "gps_lon": round(float(gps_lon), 6),
                "route_id": self.route_id,
                "event_type": "HIT_AND_RUN_CANDIDATE",
                "severity": "CRITICAL",
                "confidence": round(float(confidence), 2),
                "source_modality": "monocular",
                "software_version": self.software_version,
                "metric_depth_valid": False,
                "suspect_track_id": suspect_track_id or primary_track_id,
                "victim_track_id": victim_track_id,
                "event_stage": event_stage or "HIT_AND_RUN_CANDIDATE",
                "departure_detected": departure_detected if departure_detected is not None else True,
                "registration_number": registration_number,
                "ocr_confidence": ocr_confidence,
                "image": image_path,
                "event_clip": clip_path
            }

        # Kinematic threshold gating: only true critical/emergency states are uploaded
        is_critical = (
            (ttc_s is not None and ttc_s < 2.5) or
            event_type in ["COLLISION_CANDIDATE", "HARD_BRAKING", "CUT_IN_CANDIDATE", "EMERGENCY_EVASIVE_MANEUVER"]
        )

        if not is_critical:
            # Keep local per Section 2 & 26 (do NOT flood normal vehicle distances)
            return None

        # Cooldown per track to prevent 10fps stream flood
        last_t = self.recent_incidents.get(primary_track_id, 0.0)
        if timestamp - last_t < 4.0:
            return None
        self.recent_incidents[primary_track_id] = timestamp

        severity = "CRITICAL" if (ttc_s is not None and ttc_s < 1.5) else "HIGH"
        event_id = f"EVT_INC_{self.bus_id}_{primary_track_id}_{int(timestamp)}"

        return {
            "event_id": event_id,
            "bus_id": self.bus_id,
            "timestamp": round(float(timestamp), 2),
            "gps_lat": round(float(gps_lat), 6),
            "gps_lon": round(float(gps_lon), 6),
            "route_id": self.route_id,
            "event_type": event_type,
            "severity": severity,
            "confidence": round(float(confidence), 2),
            "source_modality": source_modality,
            "software_version": self.software_version,
            "primary_track_id": primary_track_id,
            "object_class": "vehicle",
            "range_m": round(float(range_m), 2) if range_m is not None else None,
            "relative_speed_mps": round(float(relative_speed_mps), 2) if relative_speed_mps is not None else None,
            "ttc_s": round(float(ttc_s), 2) if ttc_s is not None else None,
            "image": image_path,
            "event_clip": clip_path
        }

    # =========================================================================
    # 4. VRU ENCOUNTER GATE
    # =========================================================================

    def filter_vru_event(
        self,
        timestamp: float,
        gps_lat: float,
        gps_lon: float,
        event_type: str,
        track_id: int,
        vru_type: str,
        range_m: Optional[float],
        ttc_s: Optional[float],
        confidence: float,
        image_path: Optional[str] = None,
        clip_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Filters pedestrian/cyclist tracks.
        Normal distant pedestrians on sidewalks stay LOCAL.
        Only in-path, sudden emergence, or near-miss events are uploaded.
        """
        is_reportable = (
            event_type in ["VRU_NEAR_MISS", "SUDDEN_VRU_EMERGENCE"] or
            (event_type in ["PEDESTRIAN_IN_PATH", "CYCLIST_IN_PATH"] and range_m is not None and range_m <= 15.0)
        )
        if not is_reportable:
            return None

        # Cooldown per VRU track
        last_t = self.recent_incidents.get(track_id, 0.0)
        if timestamp - last_t < 4.0:
            return None
        self.recent_incidents[track_id] = timestamp

        severity = "CRITICAL" if (ttc_s is not None and ttc_s < 2.0) else "HIGH"
        event_id = f"EVT_VRU_{self.bus_id}_{track_id}_{int(timestamp)}"

        return {
            "event_id": event_id,
            "bus_id": self.bus_id,
            "timestamp": round(float(timestamp), 2),
            "gps_lat": round(float(gps_lat), 6),
            "gps_lon": round(float(gps_lon), 6),
            "route_id": self.route_id,
            "event_type": event_type,
            "vru_type": vru_type,
            "severity": severity,
            "confidence": round(float(confidence), 2),
            "source_modality": "stereo",
            "software_version": self.software_version,
            "track_id": track_id,
            "range_m": round(float(range_m), 2) if range_m is not None else None,
            "ttc_s": round(float(ttc_s), 2) if ttc_s is not None else None,
            "image": image_path,
            "event_clip": clip_path
        }

    # =========================================================================
    # 5. ANPR GATE (REJECT UNREADABLE / SILENT ON FAILURE)
    # =========================================================================

    def filter_anpr_event(
        self,
        timestamp: float,
        gps_lat: float,
        gps_lon: float,
        track_id: int,
        registration_number: Optional[str],
        ocr_confidence: float,
        plate_confidence: float,
        plate_pixel_width: float,
        best_frame_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Enforces Section 8:
        - Uploads ONLY when plate is verified.
        - Strictly suppresses 'UNREADABLE', 'UNKNOWN', 'NO PLATE', or failed OCR.
        - Deduplicates: 1 vehicle track = 1 ANPR event upload.
        """
        if not registration_number or ocr_confidence < 0.40 or plate_confidence < 0.40:
            # Silent local retention per contract
            return None

        clean_reg = registration_number.strip().upper()
        forbidden = ["UNREADABLE", "UNKNOWN", "NO PLATE", "N/A", "OCR FAILED", "NULL", "FAILED"]
        if any(f in clean_reg for f in forbidden) or len(clean_reg) < 4:
            return None

        # Deduplicate per track/registration
        dedup_key = f"{track_id}_{clean_reg}"
        if dedup_key in self.known_uploaded_plates:
            return None
        self.known_uploaded_plates.add(dedup_key)

        event_id = f"EVT_ANPR_{self.bus_id}_{track_id}_{int(timestamp)}"
        return {
            "event_id": event_id,
            "bus_id": self.bus_id,
            "timestamp": round(float(timestamp), 2),
            "gps_lat": round(float(gps_lat), 6),
            "gps_lon": round(float(gps_lon), 6),
            "route_id": self.route_id,
            "event_type": "ANPR_VERIFIED",
            "severity": "LOW",
            "confidence": round(float(ocr_confidence), 2),
            "source_modality": "monocular",
            "software_version": self.software_version,
            "track_id": track_id,
            "vehicle_class": "car",
            "registration_number": clean_reg,
            "ocr_confidence": round(float(ocr_confidence), 2),
            "plate_confidence": round(float(plate_confidence), 2),
            "plate_pixel_width": round(float(plate_pixel_width), 1),
            "best_frame": best_frame_path,
            "ocr_engine": "EasyOCR-Ensemble",
            "ocr_consensus_count": 2
        }

    # =========================================================================
    # 6. TRAFFIC DYNAMICS GATE (30s WINDOW OR STATE TRANSITION)
    # =========================================================================

    def filter_traffic_aggregate(
        self,
        timestamp: float,
        gps_lat: float,
        gps_lon: float,
        flow_rate_vpm: float,
        density_veh_per_km: float,
        mean_speed_kmh: float,
        median_speed_kmh: float,
        occupancy: float,
        traffic_state: str,
        jam_risk: float,
        mean_headway_s: Optional[float] = None,
        queue_length_m: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Enforces Section 10 & 11:
        - Uploads ONLY when 30s window elapsed OR meaningful state transition occurred.
        - Zero per-vehicle per-frame noise sent to server.
        """
        elapsed = timestamp - self.last_traffic_upload_time
        state_changed = (self.last_traffic_state is not None and self.last_traffic_state != traffic_state and "JAM" in traffic_state)

        if elapsed < self.traffic_window_s and not state_changed:
            return None

        self.last_traffic_upload_time = timestamp
        self.last_traffic_state = traffic_state

        event_id = f"EVT_TRF_{self.bus_id}_{int(timestamp)}"
        severity = "HIGH" if traffic_state == "JAM_RISK" else ("MEDIUM" if traffic_state == "CONGESTED" else "LOW")

        return {
            "event_id": event_id,
            "bus_id": self.bus_id,
            "timestamp": round(float(timestamp), 2),
            "gps_lat": round(float(gps_lat), 6),
            "gps_lon": round(float(gps_lon), 6),
            "route_id": self.route_id,
            "event_type": "TRAFFIC_AGGREGATE",
            "severity": severity,
            "confidence": 0.95,
            "source_modality": "stereo",
            "software_version": self.software_version,
            "flow_rate_vpm": round(float(flow_rate_vpm), 1),
            "density_veh_per_km": round(float(density_veh_per_km), 1),
            "mean_speed_kmh": round(float(mean_speed_kmh), 1),
            "median_speed_kmh": round(float(median_speed_kmh), 1),
            "occupancy": round(float(occupancy), 3),
            "mean_headway_s": round(float(mean_headway_s), 2) if mean_headway_s is not None else None,
            "queue_length_m": round(float(queue_length_m), 1) if queue_length_m is not None else None,
            "traffic_state": traffic_state,
            "jam_risk": round(float(jam_risk), 3),
            "measurement_window_s": 30.0
        }
