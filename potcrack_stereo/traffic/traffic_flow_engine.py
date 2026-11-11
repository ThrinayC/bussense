#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Vehicular Perception: Traffic Flow Engine & Congestion/Jam-Risk Predictor
=====================================================================================
Calculates physically grounded traffic flow metrics and predictive jam risk:
1. Flow Rate (veh/min, veh/hr) via virtual measurement line / gate crossing in ego road corridor.
2. Vehicle Density (veh/km) over observable metric road segment [Zmin, Zmax].
3. Metric Speeds: mean, median, speed variance, 85th percentile (km/h & m/s).
4. Spatial/Time Occupancy: fraction of observable road polygon occupied by vehicular footprints.
5. Time Headway: mean time headway and minimum headway (s).
6. Queue Length: metric longitudinal extent of sustained stationary/slow vehicles (m).
7. Flow Trends & Jam Risk Prediction:
   - Evaluates short-term trends (dDensity/dt, dFlow/dt, dSpeed/dt, dQueue/dt)
   - Classifies state: FREE FLOW | BUILDING | CONGESTED | JAM RISK
   - Grounded strictly in tracked 3D trajectories from real stereo perception.
"""

from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class TrafficFlowEngine:
    """
    Advanced traffic engineering measurement layer operating above 3D vehicle tracking.
    """

    def __init__(
        self,
        gate_distance_m: float = 15.0,
        gate_half_width_m: float = 3.6,
        z_min: float = 2.0,
        z_max: float = 35.0,
        road_width_m: float = 7.0,
        window_sec: float = 30.0,
        fps: float = 10.0,
    ):
        """
        Args:
            gate_distance_m: Longitudinal distance Z of virtual loop detector gate.
            gate_half_width_m: Lateral window |X| for gate crossings.
            z_min, z_max: Observable 3D range in meters.
            road_width_m: Assumed road corridor width in meters.
            window_sec: Temporal sliding window for flow rate estimation.
            fps: Frame rate for timestamp derivation when missing.
        """
        self.gate_distance_m = float(gate_distance_m)
        self.gate_half_width_m = float(gate_half_width_m)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.road_width_m = float(road_width_m)
        self.window_sec = float(window_sec)
        self.fps = float(fps)
        self.road_segment_length_km = max(0.001, (self.z_max - self.z_min) / 1000.0)

        # Gate crossing tracking: track_id -> previous Z
        self._prev_track_z: Dict[int, float] = {}
        # Timestamps of crossing events: deque of timestamps
        self._gate_crossings: deque = deque()

        # Rolling history of metric snapshots for trend analysis
        self._history_snapshots: deque = deque(maxlen=int(window_sec * fps))

        # Smoothed state
        self._last_state: str = "FREE FLOW"
        self._last_jam_risk: float = 0.05

    def reset(self):
        """Clears temporal state."""
        self._prev_track_z.clear()
        self._gate_crossings.clear()
        self._history_snapshots.clear()
        self._last_state = "FREE FLOW"
        self._last_jam_risk = 0.05

    def update(
        self,
        tracks: List[Any],
        timestamp: float,
        ego_speed_mps: float = 0.0,
        road_polygon_area_m2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Updates flow calculations given active 3D tracks at current timestamp.

        Args:
            tracks: List of tracked objects with current_position_3d (X, Z), current_velocity_3d (vx, vz),
                    and track_id.
            timestamp: Current timestamp in seconds.
            ego_speed_mps: Ego vehicle speed for world-frame velocity compensation.
            road_polygon_area_m2: Observable road polygon area in m² (if available).

        Returns:
            Dict containing complete traffic flow metrics and predictive jam risk state.
        """
        # Filter vehicle tracks within observable road segment
        veh_tracks = []
        for t in tracks:
            cat = getattr(t, "category", "")
            cls_name = getattr(t, "stable_class", getattr(t, "cls_name", ""))
            if cat == "vehicle" or cls_name in ["car", "truck", "bus", "van"]:
                pos = getattr(t, "current_position_3d", None)
                if pos is not None:
                    x, z = pos[0], pos[1]
                    if self.z_min <= z <= self.z_max and abs(x) <= self.road_width_m:
                        veh_tracks.append(t)

        current_veh_count = len(veh_tracks)

        # -------------------------------------------------------------
        # 1. GATE CROSSING & FLOW RATE
        # -------------------------------------------------------------
        current_active_ids = set()
        for t in veh_tracks:
            tid = t.track_id
            current_active_ids.add(tid)
            x, z = t.current_position_3d
            prev_z = self._prev_track_z.get(tid)
            if prev_z is not None and abs(x) <= self.gate_half_width_m:
                # Crossed gate from far to near (z > gate >= prev_z or prev_z > gate >= z)
                if (prev_z > self.gate_distance_m >= z) or (prev_z >= self.gate_distance_m > z):
                    self._gate_crossings.append(timestamp)
                elif (prev_z < self.gate_distance_m <= z) or (prev_z <= self.gate_distance_m < z):
                    self._gate_crossings.append(timestamp)
            self._prev_track_z[tid] = z

        # Prune old tracks no longer seen
        dead_ids = [tid for tid in self._prev_track_z if tid not in current_active_ids]
        for tid in dead_ids:
            del self._prev_track_z[tid]

        # Prune crossing events older than window_sec
        cutoff_ts = timestamp - self.window_sec
        while self._gate_crossings and self._gate_crossings[0] < cutoff_ts:
            self._gate_crossings.popleft()

        # Compute flow rate
        active_window_duration = max(1.0, min(self.window_sec, timestamp if timestamp > 0 else self.window_sec))
        num_crossings = len(self._gate_crossings)
        # Flow rate in vehicles per minute & vehicles per hour
        flow_rate_vpm = (num_crossings / active_window_duration) * 60.0
        flow_rate_vph = flow_rate_vpm * 60.0

        # Fallback/baseline estimation when sequence is short (< gate window)
        # Standard fundamental diagram: Flow q = k * v
        # If no gate crossings yet but vehicles are moving through:
        if flow_rate_vpm < 1.0 and current_veh_count > 0:
            avg_v_raw = np.mean([abs(getattr(t, "current_velocity_3d", (0.0, 0.0))[1]) + ego_speed_mps for t in veh_tracks])
            density_proxy_km = current_veh_count / self.road_segment_length_km
            q_proxy_vph = density_proxy_km * (avg_v_raw * 3.6)
            flow_rate_vpm = round(float(np.clip(q_proxy_vph / 60.0, 0.0, 120.0)), 1)
            flow_rate_vph = round(flow_rate_vpm * 60.0, 1)
        else:
            flow_rate_vpm = round(float(flow_rate_vpm), 1)
            flow_rate_vph = round(float(flow_rate_vph), 1)

        # -------------------------------------------------------------
        # 2. VEHICLE DENSITY (veh/km)
        # -------------------------------------------------------------
        density_veh_per_km = round(float(current_veh_count / self.road_segment_length_km), 1)

        # -------------------------------------------------------------
        # 3. VEHICLE SPEEDS (Mean, Median, Variance, 85th Percentile)
        # -------------------------------------------------------------
        speeds_mps = []
        for t in veh_tracks:
            vx, vz = getattr(t, "current_velocity_3d", (0.0, 0.0))
            # Relative speed + ego speed compensation for forward movement
            v_ego_comp = abs(vz) + (ego_speed_mps if ego_speed_mps > 0 else 0.0)
            total_speed = np.sqrt(vx**2 + v_ego_comp**2)
            speeds_mps.append(total_speed)

        if speeds_mps:
            speeds_kmh = [s * 3.6 for s in speeds_mps]
            mean_speed_mps = round(float(np.mean(speeds_mps)), 2)
            mean_speed_kmh = round(float(np.mean(speeds_kmh)), 1)
            median_speed_kmh = round(float(np.median(speeds_kmh)), 1)
            speed_variance_kmh2 = round(float(np.var(speeds_kmh)), 2)
            speed_p85_kmh = round(float(np.percentile(speeds_kmh, 85)), 1)
        else:
            mean_speed_mps = round(float(ego_speed_mps), 2)
            mean_speed_kmh = round(float(ego_speed_mps * 3.6), 1)
            median_speed_kmh = mean_speed_kmh
            speed_variance_kmh2 = 0.0
            speed_p85_kmh = mean_speed_kmh

        # -------------------------------------------------------------
        # 4. OCCUPANCY (Physical road polygon fraction)
        # -------------------------------------------------------------
        # Standard passenger car footprint ~ 4.5m x 1.8m = 8.1 m²
        avg_footprint_m2 = 8.5
        total_veh_footprint_m2 = current_veh_count * avg_footprint_m2
        eff_road_area = road_polygon_area_m2 if (road_polygon_area_m2 and road_polygon_area_m2 > 10.0) else (self.road_segment_length_km * 1000.0 * self.road_width_m)
        occupancy = round(float(np.clip(total_veh_footprint_m2 / eff_road_area, 0.0, 1.0)), 3)

        # -------------------------------------------------------------
        # 5. TIME HEADWAY (Mean & Minimum Headway)
        # -------------------------------------------------------------
        headways = []
        # Sort vehicles longitudinally
        sorted_vehs = sorted(veh_tracks, key=lambda t: t.current_position_3d[1])
        for i in range(len(sorted_vehs) - 1):
            dz = sorted_vehs[i + 1].current_position_3d[1] - sorted_vehs[i].current_position_3d[1]
            # Speed of following vehicle
            v_follow = max(1.0, speeds_mps[i + 1] if (i + 1 < len(speeds_mps)) else 5.0)
            th = dz / v_follow
            if 0.1 <= th <= 15.0:
                headways.append(th)

        if headways:
            mean_headway_s = round(float(np.mean(headways)), 2)
            min_headway_s = round(float(np.min(headways)), 2)
        else:
            mean_headway_s = 2.5
            min_headway_s = 2.0

        # -------------------------------------------------------------
        # 6. QUEUE LENGTH (Spatial extent of sustained slow/stationary vehicles)
        # -------------------------------------------------------------
        slow_vehs = []
        for i, t in enumerate(veh_tracks):
            sp = speeds_mps[i] if i < len(speeds_mps) else 0.0
            if sp < (10.0 / 3.6):  # Below 10 km/h is slow/queued
                slow_vehs.append(t.current_position_3d[1])

        if len(slow_vehs) >= 2:
            queue_length_m = round(float(max(slow_vehs) - min(slow_vehs)), 1)
        elif len(slow_vehs) == 1:
            queue_length_m = 5.0  # single queued vehicle footprint
        else:
            queue_length_m = 0.0

        # -------------------------------------------------------------
        # 7. FLOW TREND & CONGESTION / JAM RISK PREDICTION
        # -------------------------------------------------------------
        snapshot = {
            "timestamp": timestamp,
            "density": density_veh_per_km,
            "flow": flow_rate_vpm,
            "speed": mean_speed_kmh,
            "queue": queue_length_m,
            "occupancy": occupancy,
        }
        self._history_snapshots.append(snapshot)

        # Compute trends if we have sufficient history (>= 1.0s / 10 frames)
        if len(self._history_snapshots) >= 10:
            earlier = self._history_snapshots[0]
            dt_trend = max(0.5, timestamp - earlier["timestamp"])
            d_density = (density_veh_per_km - earlier["density"]) / dt_trend
            d_flow = (flow_rate_vpm - earlier["flow"]) / dt_trend
            d_speed = (mean_speed_kmh - earlier["speed"]) / dt_trend
            d_queue = (queue_length_m - earlier["queue"]) / dt_trend
        else:
            d_density, d_flow, d_speed, d_queue = 0.0, 0.0, 0.0, 0.0

        # Jam risk score synthesis [0.0, 1.0]
        # Indicators:
        # - High density (>45 veh/km)
        # - Low mean speed (<20 km/h)
        # - Queue presence (>10m)
        # - High occupancy (>0.25)
        # - Low headway (<1.2s)
        # - Dynamic trend: d_density > 0 and d_speed < 0 and d_flow < 0
        jam_score = 0.0
        if density_veh_per_km > 30.0:
            jam_score += 0.20 * min(1.0, (density_veh_per_km - 30.0) / 40.0)
        if mean_speed_kmh < 30.0:
            jam_score += 0.25 * min(1.0, (30.0 - mean_speed_kmh) / 25.0)
        if queue_length_m > 5.0:
            jam_score += 0.25 * min(1.0, queue_length_m / 25.0)
        if occupancy > 0.15:
            jam_score += 0.15 * min(1.0, (occupancy - 0.15) / 0.25)
        if min_headway_s < 1.5:
            jam_score += 0.10 * min(1.0, (1.5 - min_headway_s) / 1.0)

        # Dynamic trend exacerbation
        if d_density > 0.5 and d_speed < -0.5:
            jam_score += 0.15
        if d_queue > 0.5:
            jam_score += 0.10

        jam_risk = float(np.clip(jam_score, 0.0, 1.0))
        # Smooth jam risk
        self._last_jam_risk = 0.8 * self._last_jam_risk + 0.2 * jam_risk

        # State determination
        if current_veh_count == 0:
            traffic_state = "FREE FLOW"
        elif self._last_jam_risk > 0.65 or (queue_length_m > 20.0 and mean_speed_kmh < 15.0):
            traffic_state = "JAM RISK"
        elif density_veh_per_km > 40.0 or occupancy > 0.22 or (mean_speed_kmh < 22.0 and current_veh_count > 0):
            traffic_state = "CONGESTED"
        elif density_veh_per_km > 20.0 or flow_rate_vpm > 30.0:
            traffic_state = "BUILDING"
        else:
            traffic_state = "FREE FLOW"

        self._last_state = traffic_state

        return {
            "flow_rate_vpm": flow_rate_vpm,
            "flow_rate_vph": flow_rate_vph,
            "density_veh_per_km": density_veh_per_km,
            "mean_speed_mps": mean_speed_mps,
            "mean_speed_kmh": mean_speed_kmh,
            "median_speed_kmh": median_speed_kmh,
            "speed_variance_kmh2": speed_variance_kmh2,
            "speed_p85_kmh": speed_p85_kmh,
            "occupancy": occupancy,
            "mean_headway_s": mean_headway_s,
            "min_headway_s": min_headway_s,
            "queue_length_m": queue_length_m,
            "traffic_state": traffic_state,
            "jam_risk": round(self._last_jam_risk, 3),
            "trends": {
                "d_density_dt": round(d_density, 3),
                "d_flow_dt": round(d_flow, 3),
                "d_speed_dt": round(d_speed, 3),
                "d_queue_dt": round(d_queue, 3),
            },
            "vehicles_observed": current_veh_count,
        }
