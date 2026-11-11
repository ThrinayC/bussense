"""
Vulnerable Road Area Aggregation Engine — Phase 6
Aggregates spatial VRU observations across bus passes into repeated hotspot layers
and generates GIS-compatible GeoJSON outputs per PRD §21, §22, §50.
"""

import json
from typing import Dict, List, Optional
import numpy as np


class VulnerableAreaEngine:
    """
    Spatially clusters VRU observations and calculates the BusSense Vulnerable Area Index.
    """

    def __init__(self, min_observations_threshold: int = 2, grid_size_deg: float = 0.0002):
        # Grid size approx 20 meters in latitude/longitude
        self.min_observations_threshold = min_observations_threshold
        self.grid_size_deg = grid_size_deg
        self.observations: List[Dict] = []

    def record_observation(
        self,
        lat: float,
        lon: float,
        timestamp: float,
        vru_class: str,
        crossing: bool,
        in_path: bool,
        near_miss: bool,
        small_stature_candidate: bool,
        distance_m: float,
        ttc_s: Optional[float] = None,
        frame_id: int = 0,
    ) -> Dict:
        """Records a single spatial observation per PRD §48."""
        obs_id = f"VRU_OBS_{len(self.observations) + 1:05d}"
        
        # Grid segment quantization
        grid_lat = round(lat / self.grid_size_deg) * self.grid_size_deg
        grid_lon = round(lon / self.grid_size_deg) * self.grid_size_deg
        segment_id = f"SEG_{grid_lat:.5f}_{grid_lon:.5f}"

        obs = {
            "observation_id": obs_id,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "gps": {"lat": round(lat, 6), "lon": round(lon, 6)},
            "road_segment_id": segment_id,
            "vru_class": vru_class,
            "crossing": crossing,
            "in_path": in_path,
            "near_miss": near_miss,
            "small_stature_candidate": small_stature_candidate,
            "distance_m": distance_m,
            "ttc_s": ttc_s,
        }
        self.observations.append(obs)
        return obs

    def aggregate_segments(self, bus_pass_count: int = 1) -> Dict[str, Dict]:
        """
        Groups observations by spatial segment and calculates exposure-normalized metrics.
        """
        segments: Dict[str, List[Dict]] = {}
        for obs in self.observations:
            seg_id = obs["road_segment_id"]
            if seg_id not in segments:
                segments[seg_id] = []
            segments[seg_id].append(obs)

        aggregated: Dict[str, Dict] = {}

        for seg_id, obs_list in segments.items():
            n = len(obs_list)
            lats = [o["gps"]["lat"] for o in obs_list]
            lons = [o["gps"]["lon"] for o in obs_list]

            ped_count = sum(1 for o in obs_list if o["vru_class"] == "person")
            cyc_count = sum(1 for o in obs_list if o["vru_class"] == "bicycle")
            crossing_count = sum(1 for o in obs_list if o["crossing"])
            in_path_count = sum(1 for o in obs_list if o["in_path"])
            near_miss_count = sum(1 for o in obs_list if o["near_miss"])
            small_stature_count = sum(1 for o in obs_list if o["small_stature_candidate"])

            # BusSense Vulnerable Area Index (VAI):
            # VAI = [crossings * 1.0 + in_path * 2.0 + near_misses * 5.0] / bus_passes
            raw_risk = (crossing_count * 1.0) + (in_path_count * 2.0) + (near_miss_count * 5.0)
            vai_score = round(float(raw_risk / max(1, bus_pass_count)), 2)

            is_repeated_hotspot = n >= self.min_observations_threshold

            aggregated[seg_id] = {
                "segment_id": seg_id,
                "center_gps": {"lat": round(float(np.mean(lats)), 6), "lon": round(float(np.mean(lons)), 6)},
                "observation_count": n,
                "bus_pass_count": bus_pass_count,
                "pedestrian_count": ped_count,
                "cyclist_count": cyc_count,
                "crossing_count": crossing_count,
                "in_path_count": in_path_count,
                "near_miss_count": near_miss_count,
                "small_stature_candidate_count": small_stature_count,
                "bussense_vai_score": vai_score,
                "is_repeated_hotspot": is_repeated_hotspot,
            }

        return aggregated

    def export_geojson(self, aggregated: Dict[str, Dict], out_path: str):
        """Exports aggregated spatial clusters to GeoJSON format."""
        features = []
        for seg_id, data in aggregated.items():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [data["center_gps"]["lon"], data["center_gps"]["lat"]],
                },
                "properties": {
                    "segment_id": seg_id,
                    "observation_count": data["observation_count"],
                    "bus_pass_count": data["bus_pass_count"],
                    "crossing_count": data["crossing_count"],
                    "in_path_count": data["in_path_count"],
                    "near_miss_count": data["near_miss_count"],
                    "vai_score": data["bussense_vai_score"],
                    "repeated_hotspot": data["is_repeated_hotspot"],
                },
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "title": "BusSense Vulnerable Area Layer",
                "total_segments": len(aggregated),
                "total_observations": len(self.observations),
            },
            "features": features,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)
