#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/trichy_replay_simulator.py
==================================================
Simulated Multi-Bus Replay & Synthetic Fault Injection Runner (Sections 24, 25, 26, 27)

Implements:
1. Ingestion of official demo reference assets (Section 15, 24).
2. BUS_DEMO_001 pass with synthetic divider fault at 30m-45m -> MISSING_DIVIDER_CANDIDATE.
3. BUS_DEMO_002 and BUS_DEMO_003 passes -> merged into 1 cluster with unique_bus_count = 3 (CORROBORATED).
4. BUS_DEMO_004 conflicting pass (divider present) -> CONFLICTING_EVIDENCE retained.
5. Full continuous telemetry recording adhering to distance_to_centerline_m <= 10m.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.trichy_road_geometry import TrichyRoadGeometryManager
from bussense_infrastructure.road_divider_engine import RoadDividerEngine
from bussense_infrastructure.evidence_engine import EvidenceEngine
from bussense_infrastructure.multi_bus_fusion import MultiBusFusionEngine
from bussense_server.database import BusSenseDatabase


class TrichyReplaySimulator:
    """
    Orchestrates the road-constrained Trichy multi-bus replay and synthetic fault injection.
    """

    def __init__(self, db: Optional[BusSenseDatabase] = None):
        self.db = db or BusSenseDatabase()
        self.geom_mgr = TrichyRoadGeometryManager()
        self.divider_engine = RoadDividerEngine()
        self.evidence_engine = EvidenceEngine()
        self.fusion_engine = MultiBusFusionEngine()

    def setup_reference_assets(self):
        """Seeds the manual demo reference assets into the database."""
        # Expected divider on Bharathidasan Salai
        ref_divider = {
            "asset_id": "REF-DIVIDER-TRICHY-01",
            "road_segment_id": self.geom_mgr.ROAD_SEGMENT_ID,
            "asset_type": "ROAD_DIVIDER",
            "asset_subtype": "CONCRETE_JERSEY_MEDIAN",
            "expected_state": "YES",
            "reference_source": "MANUAL_DEMO_REFERENCE",
            "source_quality": "DEMO",
            "geometry": {
                "road_name": self.geom_mgr.ROAD_NAME,
                "osm_way_id": self.geom_mgr.OSM_WAY_ID,
                "start_m": 0.0,
                "end_m": round(self.geom_mgr.total_road_length_m, 2)
            },
            "last_verified": "2026-09-28"
        }
        self.db.insert_reference_asset(ref_divider)

        # Expected STOP sign at intersection
        ref_sign = {
            "asset_id": "REF-SIGN-TRICHY-01",
            "road_segment_id": self.geom_mgr.ROAD_SEGMENT_ID,
            "asset_type": "TRAFFIC_SIGN",
            "asset_subtype": "STOP",
            "expected_state": "YES",
            "reference_source": "MANUAL_DEMO_REFERENCE",
            "source_quality": "DEMO",
            "geometry": {"distance_along_road_m": 120.0, "latitude": 10.8035, "longitude": 78.6878},
            "allowed_openings": [],
            "last_verified": "2026-09-28"
        }
        self.db.insert_reference_asset(ref_sign)

        # Expected Zebra crossing
        ref_zebra = {
            "asset_id": "REF-ZEBRA-TRICHY-01",
            "road_segment_id": self.geom_mgr.ROAD_SEGMENT_ID,
            "asset_type": "ZEBRA_CROSSING",
            "asset_subtype": "STANDARD_ZEBRA",
            "expected_state": "YES",
            "reference_source": "MANUAL_DEMO_REFERENCE",
            "source_quality": "DEMO",
            "geometry": {"distance_along_road_m": 250.0, "latitude": 10.8049, "longitude": 78.6857},
            "allowed_openings": [],
            "last_verified": "2026-09-28"
        }
        self.db.insert_reference_asset(ref_zebra)

    def run_simulation(self) -> Dict[str, Any]:
        """
        Executes the full multi-bus simulation:
        - BUS_DEMO_001: Sees synthetic gap at 30m-45m -> CANDIDATE.
        - BUS_DEMO_002: Sees same gap -> CORROBORATED (buses=2).
        - BUS_DEMO_003: Sees same gap -> CORROBORATED (buses=3).
        - BUS_DEMO_004: Conflicting pass -> CONFLICTING_EVIDENCE.
        """
        self.setup_reference_assets()
        sim_start_time = time.time()

        # -------------------------------------------------------------
        # PASS 1: BUS_DEMO_001
        # -------------------------------------------------------------
        traj_001 = self.geom_mgr.generate_full_bus_trajectory("BUS_DEMO_001", speed_kmh=25.0, start_time=sim_start_time)
        obs_bus1 = []
        for p in traj_001:
            self.db.record_bus_telemetry(p)
            dist_m = p["distance_along_road_m"]

            # Synthetic fault: 30m -> 45m missing divider
            is_fault_zone = (30.0 <= dist_m <= 45.0)

            if is_fault_zone:
                obs_data = {
                    "observation_id": f"OBS-B1-FAULT-{int(dist_m)}",
                    "bus_id": "BUS_DEMO_001",
                    "timestamp": p["timestamp"],
                    "latitude": p["latitude"],
                    "longitude": p["longitude"],
                    "heading": p["heading"],
                    "road_segment_id": p["road_segment_id"],
                    "event_type": "MISSING_DIVIDER_CANDIDATE",
                    "model_confidence": 0.92,
                    "visibility_score": 0.94,
                    "parameters": {
                        "is_missing": True,
                        "gap_start_m": 30.0,
                        "gap_end_m": 45.0,
                        "gap_length_m": 15.0,
                        "simulation_mode": True,
                        "observation_mode": "SIMULATED_MISSING_ASSET"
                    },
                    "simulation_mode": True
                }
                self.db.insert_observation(obs_data)
                obs_bus1.append(obs_data)

        # Ingest Pass 1 into Multi-Bus Fusion Engine
        clusters = self.fusion_engine.ingest_bus_pass("BUS_DEMO_001", "PASS-01", obs_bus1, simulation_mode=True)

        # -------------------------------------------------------------
        # PASS 2 & 3: BUS_DEMO_002 & BUS_DEMO_003 (Corroboration)
        # -------------------------------------------------------------
        for b_name, p_name in [("BUS_DEMO_002", "PASS-02"), ("BUS_DEMO_003", "PASS-03")]:
            traj = self.geom_mgr.generate_full_bus_trajectory(b_name, speed_kmh=26.0, start_time=sim_start_time + 30.0)
            b_obs = []
            for p in traj:
                self.db.record_bus_telemetry(p)
                dist_m = p["distance_along_road_m"]
                if 30.0 <= dist_m <= 45.0:
                    o = {
                        "observation_id": f"OBS-{b_name}-{int(dist_m)}",
                        "bus_id": b_name,
                        "timestamp": p["timestamp"],
                        "latitude": p["latitude"],
                        "longitude": p["longitude"],
                        "heading": p["heading"],
                        "road_segment_id": p["road_segment_id"],
                        "event_type": "MISSING_DIVIDER_CANDIDATE",
                        "model_confidence": 0.91,
                        "visibility_score": 0.93,
                        "parameters": {
                            "is_missing": True,
                            "gap_start_m": 30.0,
                            "gap_end_m": 45.0,
                            "gap_length_m": 15.0,
                            "simulation_mode": True
                        },
                        "simulation_mode": True
                    }
                    self.db.insert_observation(o)
                    b_obs.append(o)

            clusters = self.fusion_engine.ingest_bus_pass(b_name, p_name, b_obs, simulation_mode=True)

        # State after 3 agreeing buses
        cluster_3buses = self.fusion_engine.get_cluster("CLUST-MISSING_DIVIDER_CANDIDATE-0001")
        corroborated_status = cluster_3buses.status if cluster_3buses else "UNKNOWN"
        corroborated_bus_count = len(cluster_3buses.unique_buses) if cluster_3buses else 0

        # -------------------------------------------------------------
        # PASS 4: BUS_DEMO_004 (Conflicting Pass: Divider Present)
        # -------------------------------------------------------------
        traj_004 = self.geom_mgr.generate_full_bus_trajectory("BUS_DEMO_004", speed_kmh=24.0, start_time=sim_start_time + 90.0)
        obs_bus4 = []
        for p in traj_004:
            self.db.record_bus_telemetry(p)
            dist_m = p["distance_along_road_m"]
            if 30.0 <= dist_m <= 45.0:
                o = {
                    "observation_id": f"OBS-B4-PRESENT-{int(dist_m)}",
                    "bus_id": "BUS_DEMO_004",
                    "timestamp": p["timestamp"],
                    "latitude": p["latitude"],
                    "longitude": p["longitude"],
                    "heading": p["heading"],
                    "road_segment_id": p["road_segment_id"],
                    "event_type": "MISSING_DIVIDER_CANDIDATE",
                    "model_confidence": 0.88,
                    "visibility_score": 0.92,
                    "parameters": {
                        "is_missing": False,  # Contradiction: Bus 4 reports divider present!
                        "divider_present": True,
                        "simulation_mode": True
                    },
                    "simulation_mode": True
                }
                self.db.insert_observation(o)
                obs_bus4.append(o)

        clusters = self.fusion_engine.ingest_bus_pass("BUS_DEMO_004", "PASS-04", obs_bus4, simulation_mode=True)

        # Sync all clusters to database infrastructure_events table
        for cl in clusters:
            ev_dict = self.fusion_engine.to_infrastructure_event_dict(cl)
            self.db.insert_or_update_infra_event(ev_dict)


        final_cluster = self.fusion_engine.get_cluster("CLUST-MISSING_DIVIDER_CANDIDATE-0001")

        return {
            "road_name": self.geom_mgr.ROAD_NAME,
            "osm_way_id": self.geom_mgr.OSM_WAY_ID,
            "road_length_m": round(self.geom_mgr.total_road_length_m, 2),
            "telemetry_points_count": len(traj_001) + len(traj_004) + 2 * len(traj_001),
            "agreeing_passes_status": corroborated_status,
            "agreeing_bus_count": corroborated_bus_count,
            "after_conflict_status": final_cluster.status if final_cluster else None,
            "total_conflicts": final_cluster.conflict_count if final_cluster else 0,
            "final_unique_bus_count": len(final_cluster.unique_buses) if final_cluster else 0,
            "simulation_mode": True
        }

    def run_real_model_replay(
        self,
        frames_dir: Optional[str] = None,
        max_frames: int = 40,
        bus_id: str = "BUS_REAL_001"
    ) -> Dict[str, Any]:
        """
        Executes real perception model inference on actual road footage (simulation_mode = False).
        Runs trained SegFormer/SegNet, TrafficSignNet, and SignConditionNet models.
        """
        import cv2
        from bussense_infrastructure.waterlogging_engine import WaterloggingEngine
        from bussense_infrastructure.traffic_sign_engine import TrafficSignEngine
        from bussense_infrastructure.zebra_crossing_engine import ZebraCrossingEngine

        water_eng = WaterloggingEngine(use_model=True)
        sign_eng = TrafficSignEngine(use_model=True)
        zebra_eng = ZebraCrossingEngine(use_model=True)
        divider_eng = self.divider_engine

        f_dir = Path(frames_dir) if frames_dir else ROOT / "datasets" / "idd" / "raw" / "images"
        if not f_dir.exists():
            f_dir = ROOT / "potcrack_stereo" / "data" / "pothole_long_stereo" / "left"

        frame_files = sorted(list(f_dir.glob("*.jpg")) + list(f_dir.glob("*.png")))[:max_frames]
        start_time = time.time()
        processed_count = 0
        real_events_count = 0

        for idx, fpath in enumerate(frame_files):
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = start_time + idx * 0.1

            # Bus telemetry point along road
            road_pt = self.geom_mgr.sample_point_at_distance(min(self.geom_mgr.total_road_length_m, idx * 8.0))
            telem = {
                "bus_id": bus_id,
                "timestamp": t,
                "latitude": road_pt[0],
                "longitude": road_pt[1],
                "speed_kmh": 32.5,
                "heading": road_pt[2],
                "accuracy_m": 1.2,
                "road_segment_id": self.geom_mgr.ROAD_SEGMENT_ID,
                "road_name": self.geom_mgr.ROAD_NAME,
                "distance_along_road_m": round(idx * 8.0, 2),
                "distance_to_centerline_m": 0.5,
                "simulation_mode": False
            }
            self.db.record_bus_telemetry(telem)

            # 1. Real Waterlogging Model
            w_obs = water_eng.process_frame(rgb, frame_idx=idx, timestamp=t)
            if w_obs and w_obs.water_coverage_ratio > 0.05:
                obs_water = {
                    "observation_id": f"OBS-REAL-WATER-{idx:04d}",
                    "bus_id": bus_id,
                    "timestamp": t,
                    "latitude": road_pt[0],
                    "longitude": road_pt[1],
                    "heading": road_pt[2],
                    "road_segment_id": self.geom_mgr.ROAD_SEGMENT_ID,
                    "event_type": "WATERLOGGING_SECTION",
                    "model_confidence": 0.88,
                    "visibility_score": 0.95,
                    "parameters": {
                        "water_coverage_ratio": w_obs.water_coverage_ratio,
                        "is_persistent": w_obs.is_persistent,
                        "observation_mode": "REAL_MODEL_INFERENCE",
                        "model_used": "LightweightRoadSegNet_Water"
                    },
                    "simulation_mode": False
                }
                self.db.insert_observation(obs_water)
                real_events_count += 1

            # 2. Real Traffic Sign Model + Condition
            signs = sign_eng.detect_candidate_signs(rgb)
            for s in signs:
                obs_sign = {
                    "observation_id": f"OBS-REAL-SIGN-{idx:04d}",
                    "bus_id": bus_id,
                    "timestamp": t,
                    "latitude": road_pt[0],
                    "longitude": road_pt[1],
                    "heading": road_pt[2],
                    "road_segment_id": self.geom_mgr.ROAD_SEGMENT_ID,
                    "event_type": "TRAFFIC_SIGN_OBSERVATION",
                    "model_confidence": s.get("confidence", 0.85),
                    "visibility_score": 0.92,
                    "parameters": {
                        "sign_type": s.get("sign_type", "OTHER"),
                        "condition": s.get("condition", "NORMAL"),
                        "condition_source": "REAL",
                        "observation_mode": "REAL_MODEL_INFERENCE",
                        "model_used": "TrafficSignNet+SignConditionNet"
                    },
                    "simulation_mode": False
                }
                self.db.insert_observation(obs_sign)
                real_events_count += 1

            # 3. Real Zebra Model
            z_env, z_stripes = zebra_eng.segment_crossing(rgb)
            if np.count_nonzero(z_env) > 500:
                obs_zebra = {
                    "observation_id": f"OBS-REAL-ZEBRA-{idx:04d}",
                    "bus_id": bus_id,
                    "timestamp": t,
                    "latitude": road_pt[0],
                    "longitude": road_pt[1],
                    "heading": road_pt[2],
                    "road_segment_id": self.geom_mgr.ROAD_SEGMENT_ID,
                    "event_type": "ZEBRA_CROSSING_OBSERVATION",
                    "model_confidence": 0.91,
                    "visibility_score": 0.95,
                    "parameters": {
                        "observation_mode": "REAL_MODEL_INFERENCE",
                        "model_used": "LightweightRoadSegNet_Zebra"
                    },
                    "simulation_mode": False
                }
                self.db.insert_observation(obs_zebra)
                real_events_count += 1

            processed_count += 1

        return {
            "bus_id": bus_id,
            "processed_frames_count": processed_count,
            "real_observations_recorded": real_events_count,
            "footage_source": str(f_dir),
            "execution_duration_sec": round(time.time() - start_time, 2),
            "simulation_mode": False,
            "perception_models_engaged": [
                "LightweightRoadSegNet_Water",
                "TrafficSignNet",
                "SignConditionNet",
                "LightweightRoadSegNet_Zebra"
            ]
        }


if __name__ == "__main__":
    sim = TrichyReplaySimulator()
    res = sim.run_simulation()
    print("[SIMULATION COMPLETE]")
    print(json.dumps(res, indent=2))
