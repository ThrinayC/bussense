"""
KITTI OXTS Ego-Motion Provider (Phase 3)
Parses synchronized GPS/IMU navigation data and provides coordinate transformations
between camera-relative, ego-relative, and world/road frames.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np


class EgoMotionFrame:
    """Represents a single-frame OXTS ego-motion record."""

    def __init__(self, raw_values: np.ndarray, timestamp: float, frame_id: int):
        self.frame_id = frame_id
        self.timestamp = timestamp

        # GPS Coordinates
        self.lat = float(raw_values[0])
        self.lon = float(raw_values[1])
        self.alt = float(raw_values[2])

        # Attitude (radians)
        self.roll = float(raw_values[3])
        self.pitch = float(raw_values[4])
        self.yaw = float(raw_values[5])

        # Velocity in nav frame (m/s)
        self.v_north = float(raw_values[6])
        self.v_east = float(raw_values[7])
        self.v_up = float(raw_values[10])

        # Velocity in vehicle body frame (m/s)
        self.v_forward = float(raw_values[8])  # Forward velocity along bus axis
        self.v_left = float(raw_values[9])     # Lateral velocity

        # Accelerations in vehicle frame (m/s^2)
        self.a_forward = float(raw_values[14])
        self.a_left = float(raw_values[15])
        self.a_up = float(raw_values[16])

        # Angular rates (rad/s)
        self.w_roll = float(raw_values[20])
        self.w_pitch = float(raw_values[21])
        self.w_yaw = float(raw_values[22])  # Yaw rate

        # Navigation status
        self.pos_acc = float(raw_values[23])
        self.vel_acc = float(raw_values[24])

    @property
    def speed_mps(self) -> float:
        return np.sqrt(self.v_forward**2 + self.v_left**2)

    @property
    def speed_kmh(self) -> float:
        return self.speed_mps * 3.6

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "speed_mps": round(self.speed_mps, 3),
            "v_forward_mps": round(self.v_forward, 3),
            "yaw_rad": round(self.yaw, 4),
            "yaw_rate_radps": round(self.w_yaw, 4),
            "a_forward_mps2": round(self.a_forward, 3),
            "pos_accuracy_m": round(self.pos_acc, 3),
        }


class EgoMotionProvider:
    """
    Loads and provides ego-motion data for a sequence.
    Handles relative vs absolute motion conversion per PRD §9 & §25.
    """

    def __init__(self, oxts_dir: str, nominal_dt: float = 0.1):
        self.oxts_dir = oxts_dir
        self.nominal_dt = nominal_dt
        self.frames: List[EgoMotionFrame] = []
        self._load_oxts_data()

    def _load_oxts_data(self):
        data_dir = os.path.join(self.oxts_dir, "data")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"OXTS data dir not found: {data_dir}")

        files = sorted(os.listdir(data_dir))
        for idx, fname in enumerate(files):
            fpath = os.path.join(data_dir, fname)
            values = np.fromfile(fpath, sep=" ")
            t = idx * self.nominal_dt
            frame = EgoMotionFrame(values, timestamp=t, frame_id=idx)
            self.frames.append(frame)

    def get_frame(self, frame_id: int) -> Optional[EgoMotionFrame]:
        if 0 <= frame_id < len(self.frames):
            return self.frames[frame_id]
        return None

    def convert_relative_to_world_kinematics(
        self,
        frame_id: int,
        rel_vz_mps: float,
        rel_vx_mps: float,
    ) -> Tuple[float, float]:
        """
        Converts camera-relative velocity (Vz, Vx) to world/road-relative target velocity
        using ego bus forward velocity (vf) and lateral velocity (vl).
        In camera coordinates:
            Z is forward (+ away from bus).
            relative Vz = dZ/dt = V_target_forward - V_ego_forward
            therefore:
            V_target_forward = relative Vz + V_ego_forward.
        """
        ego = self.get_frame(frame_id)
        if ego is None:
            return rel_vz_mps, rel_vx_mps

        world_forward_v = rel_vz_mps + ego.v_forward
        world_lateral_v = rel_vx_mps - ego.v_left

        return world_forward_v, world_lateral_v

    def get_summary_metrics(self) -> Dict:
        speeds = [f.speed_mps for f in self.frames]
        yaws = [f.w_yaw for f in self.frames]
        accels = [f.a_forward for f in self.frames]

        return {
            "total_frames": len(self.frames),
            "mean_speed_mps": round(float(np.mean(speeds)), 3),
            "max_speed_mps": round(float(np.max(speeds)), 3),
            "mean_speed_kmh": round(float(np.mean(speeds) * 3.6), 2),
            "max_speed_kmh": round(float(np.max(speeds) * 3.6), 2),
            "mean_yaw_rate_radps": round(float(np.mean(np.abs(yaws))), 4),
            "max_forward_accel_mps2": round(float(np.max(accels)), 3),
            "min_forward_accel_mps2": round(float(np.min(accels)), 3),
        }
