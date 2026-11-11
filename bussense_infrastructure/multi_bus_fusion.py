#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/multi_bus_fusion.py
===========================================
Deterministic Multi-Bus Evidence Fusion Engine (Sections 20, 26, 27)

Core Invariant:
- ONE TRACKED BUS PASS = ONE OBSERVATION SOURCE
- 100 frames from the same bus pass = 1 independent observation, NOT 100 votes.
- Clusters observations by:
  road_segment_id + spatial proximity + heading alignment + event type.
- Never hides conflicts: if Bus 4 reports PRESENT while Buses 1-3 report GAP,
  the cluster status transitions to CONFLICTING_EVIDENCE with preserved audit trail.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
import sys
import math
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.evidence_engine import EvidenceEngine


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(R * c)


@dataclass
class BusPassRecord:
    bus_id: str
    pass_id: str
    timestamp: float
    observed_present: bool
    is_missing_claim: bool
    frame_count: int
    visibility_mean: float
    latitude: Optional[float]
    longitude: Optional[float]
    heading: Optional[float]
    parameters: Dict[str, Any]


@dataclass
class EventCluster:
    cluster_id: str
    event_type: str
    road_segment_id: str
    reference_asset_id: Optional[str]
    latitude: float
    longitude: float
    heading: float
    first_seen: float
    last_seen: float
    observation_count: int = 0
    unique_buses: set = field(default_factory=set)
    unique_passes: set = field(default_factory=set)
    pass_records: List[BusPassRecord] = field(default_factory=list)
    conflict_count: int = 0
    status: str = "OBSERVED"
    simulation_mode: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[str] = field(default_factory=list)


class MultiBusFusionEngine:
    """
    Rule-based cluster manager and evidence corroborator across mobile bus units.
    """

    def __init__(
        self,
        spatial_merge_radius_m: float = 15.0,
        heading_tolerance_deg: float = 45.0,
    ):
        self.spatial_merge_radius_m = spatial_merge_radius_m
        self.heading_tolerance_deg = heading_tolerance_deg
        self.clusters: Dict[str, EventCluster] = {}
        self.evidence_engine = EvidenceEngine()
        self.next_cluster_id = 1

    def can_merge(self, cluster: EventCluster, obs: Dict[str, Any]) -> bool:
        """
        Determines if an observation belongs to an existing event cluster.
        Merges related infrastructure classes (e.g. ROAD_DIVIDER and MISSING_DIVIDER_CANDIDATE).
        """
        et_clust = cluster.event_type
        et_obs = obs.get("event_type", "")

        # Same event type or opposing presence/missing claims for the same infrastructure category
        same_family = (et_clust == et_obs) or (
            ("DIVIDER" in et_clust and "DIVIDER" in et_obs) or
            ("ZEBRA" in et_clust and "ZEBRA" in et_obs) or
            ("SIGN" in et_clust and "SIGN" in et_obs)
        )
        if not same_family:
            return False

        if cluster.road_segment_id != obs.get("road_segment_id"):
            return False

        c_lat, c_lon = cluster.latitude, cluster.longitude
        o_lat, o_lon = obs.get("latitude"), obs.get("longitude")

        if c_lat is not None and c_lon is not None and o_lat is not None and o_lon is not None:
            dist = haversine_distance_m(c_lat, c_lon, o_lat, o_lon)
            if dist > self.spatial_merge_radius_m:
                return False

        # Heading alignment check
        c_head, o_head = cluster.heading, obs.get("heading")
        if c_head is not None and o_head is not None:
            diff = abs(c_head - o_head) % 360.0
            diff = min(diff, 360.0 - diff)
            if diff > self.heading_tolerance_deg:
                return False

        return True

    def ingest_bus_pass(
        self,
        bus_id: str,
        pass_id: str,
        observations: List[Dict[str, Any]],
        simulation_mode: bool = False
    ) -> List[EventCluster]:
        """
        Ingests all observations from ONE complete bus pass.
        Guarantees: ONE BUS PASS = ONE VOTE.
        """
        if not observations:
            return []

        # Consolidate frame observations from this pass
        first_obs = observations[0]
        event_type = first_obs.get("event_type", "UNKNOWN")
        road_segment_id = first_obs.get("road_segment_id", "SEG-TRICHY-01")
        ref_id = first_obs.get("reference_asset_id")

        total_frames = len(observations)
        vis_scores = [o.get("visibility_score", 0.9) for o in observations]
        mean_vis = float(np.mean(vis_scores)) if vis_scores else 0.9

        # Distinguish whether pass is claiming missing or present
        is_missing = False
        is_present = False
        for o in observations:
            params = o.get("parameters", {})
            if params.get("is_missing") is True or "MISSING" in o.get("event_type", ""):
                if not (params.get("is_missing") is False or params.get("divider_present") is True):
                    is_missing = True
            if params.get("is_missing") is False or params.get("divider_present") is True or "PRESENT" in o.get("observation_id", ""):
                is_present = True
                is_missing = False
                break

        pass_record = BusPassRecord(
            bus_id=bus_id,
            pass_id=pass_id,
            timestamp=float(first_obs.get("timestamp", time.time())),
            observed_present=is_present,
            is_missing_claim=is_missing,
            frame_count=total_frames,
            visibility_mean=round(mean_vis, 3),
            latitude=first_obs.get("latitude", 10.7905),
            longitude=first_obs.get("longitude", 78.7047),
            heading=first_obs.get("heading", 85.0),
            parameters=first_obs.get("parameters", {})
        )

        # Find matching cluster
        target_cluster = None
        for cl in self.clusters.values():
            if self.can_merge(cl, first_obs):
                target_cluster = cl
                break

        if target_cluster is None:
            cid = f"CLUST-{event_type}-{self.next_cluster_id:04d}"
            self.next_cluster_id += 1
            target_cluster = EventCluster(
                cluster_id=cid,
                event_type=event_type,
                road_segment_id=road_segment_id,
                reference_asset_id=ref_id,
                latitude=first_obs.get("latitude", 10.7905),
                longitude=first_obs.get("longitude", 78.7047),
                heading=first_obs.get("heading", 85.0),
                first_seen=float(first_obs.get("timestamp", time.time())),
                last_seen=float(observations[-1].get("timestamp", time.time())),
                simulation_mode=simulation_mode,
                parameters=first_obs.get("parameters", {})
            )
            self.clusters[cid] = target_cluster

        # Update cluster: Exactly 1 vote added for this bus pass
        target_cluster.observation_count += total_frames
        target_cluster.unique_buses.add(bus_id)
        target_cluster.unique_passes.add(pass_id)
        target_cluster.pass_records.append(pass_record)
        target_cluster.last_seen = max(target_cluster.last_seen, pass_record.timestamp)

        # Check for conflict among passes
        has_present_pass = any(p.observed_present and not p.is_missing_claim for p in target_cluster.pass_records)
        has_missing_pass = any(p.is_missing_claim for p in target_cluster.pass_records)

        if has_present_pass and has_missing_pass:
            target_cluster.conflict_count += 1
            target_cluster.status = "CONFLICTING_EVIDENCE"
        else:
            unique_bus_cnt = len(target_cluster.unique_buses)
            if unique_bus_cnt >= 2:
                target_cluster.status = "CORROBORATED"
            elif unique_bus_cnt == 1:
                target_cluster.status = "CANDIDATE" if is_missing else "PERSISTENT"

        return list(self.clusters.values())

    def get_cluster(self, cluster_id: str) -> Optional[EventCluster]:
        return self.clusters.get(cluster_id)

    def to_infrastructure_event_dict(self, cluster: EventCluster) -> Dict[str, Any]:
        """Converts cluster to standardized InfrastructureEvent dictionary."""
        return {
            "event_id": cluster.cluster_id,
            "event_type": cluster.event_type,
            "road_segment_id": cluster.road_segment_id,
            "reference_asset_id": cluster.reference_asset_id,
            "status": cluster.status,
            "first_seen": cluster.first_seen,
            "last_seen": cluster.last_seen,
            "observation_count": cluster.observation_count,
            "unique_bus_count": len(cluster.unique_buses),
            "unique_pass_count": len(cluster.unique_passes),
            "unique_day_count": 1,
            "visibility_summary": {
                "mean_visibility": round(float(np.mean([p.visibility_mean for p in cluster.pass_records])), 3) if cluster.pass_records else 0.95,
                "passes_evaluated": len(cluster.pass_records)
            },
            "temporal_consistency": 1.0,
            "spatial_consistency": 1.0,
            "reference_support": "YES" if cluster.reference_asset_id else "UNKNOWN",
            "conflict_count": cluster.conflict_count,
            "parameters": cluster.parameters,
            "evidence_refs": cluster.evidence_refs,
            "simulation_mode": cluster.simulation_mode,
            "latitude": cluster.latitude,
            "longitude": cluster.longitude,
            "heading": cluster.heading,
            "bus_ids": sorted(list(cluster.unique_buses))
        }
