"""
KITTI Calibration Parser and Coordinate Transformer
Handles Camera 0-3 Rectified Projections and Velodyne-to-Camera Calibrations.
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np


class KittiCalibration:
    """
    Parses KITTI calibration files for a synchronized raw drive or tracking sequence.
    Derives stereo baseline, focal length, fB, and rigid transformations.
    """

    def __init__(self, cam_to_cam_file: str, velo_to_cam_file: Optional[str] = None):
        self.cam_to_cam_file = cam_to_cam_file
        self.velo_to_cam_file = velo_to_cam_file
        
        if not os.path.exists(cam_to_cam_file):
            raise FileNotFoundError(f"Calibration file not found: {cam_to_cam_file}")
            
        self.raw_data: Dict[str, np.ndarray] = {}
        self._load_cam_to_cam()
        if velo_to_cam_file and os.path.exists(velo_to_cam_file):
            self._load_velo_to_cam()
            
        self._derive_stereo_parameters()
        self._build_transforms()

    def _load_cam_to_cam(self):
        with open(self.cam_to_cam_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    self.raw_data[key] = np.fromstring(val, sep=" ")
                except ValueError:
                    pass

    def _load_velo_to_cam(self):
        with open(self.velo_to_cam_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = "velo_" + key.strip()
                val = val.strip()
                try:
                    self.raw_data[key] = np.fromstring(val, sep=" ")
                except ValueError:
                    pass

    def _derive_stereo_parameters(self):
        # P_rect_02 (Camera 2 - Left Color)
        if "P_rect_02" not in self.raw_data:
            raise KeyError("P_rect_02 missing in calibration file.")
        self.P2 = self.raw_data["P_rect_02"].reshape(3, 4)

        # P_rect_03 (Camera 3 - Right Color)
        if "P_rect_03" not in self.raw_data:
            raise KeyError("P_rect_03 missing in calibration file.")
        self.P3 = self.raw_data["P_rect_03"].reshape(3, 4)

        # Intrinsic parameters from P2
        self.fx = float(self.P2[0, 0])
        self.fy = float(self.P2[1, 1])
        self.cx = float(self.P2[0, 2])
        self.cy = float(self.P2[1, 2])

        # Baseline derived from stereo projection horizontal offset:
        # P2[0, 3] = -fx * b2, P3[0, 3] = -fx * b3
        # baseline = (P2[0,3] - P3[0,3]) / fx
        self.baseline = float((self.P2[0, 3] - self.P3[0, 3]) / self.fx)
        self.fB = float(self.fx * self.baseline)

        # Image dimensions from S_rect_02 if available, else standard KITTI
        if "S_rect_02" in self.raw_data and len(self.raw_data["S_rect_02"]) == 2:
            self.image_width = int(self.raw_data["S_rect_02"][0])
            self.image_height = int(self.raw_data["S_rect_02"][1])
        else:
            self.image_width = 1242
            self.image_height = 375

    def _build_transforms(self):
        # R_rect_00 (rectification matrix for reference camera 0)
        self.R_rect_00 = np.eye(4, dtype=np.float64)
        if "R_rect_00" in self.raw_data:
            self.R_rect_00[:3, :3] = self.raw_data["R_rect_00"].reshape(3, 3)

        # Velodyne to reference camera 0
        self.Tr_velo_to_cam = np.eye(4, dtype=np.float64)
        if "velo_R" in self.raw_data and "velo_T" in self.raw_data:
            self.Tr_velo_to_cam[:3, :3] = self.raw_data["velo_R"].reshape(3, 3)
            self.Tr_velo_to_cam[:3, 3] = self.raw_data["velo_T"]

    def project_velo_to_rect_cam(self, pts_velo: np.ndarray) -> np.ndarray:
        """
        Transforms 3D points from Velodyne coordinates to Camera 2 Rectified coordinates.
        Input: (N, 3) or (N, 4) in Velodyne coordinates.
        Output: (N, 3) [X, Y, Z] in Rectified Camera coordinates.
        """
        if pts_velo.shape[1] == 4:
            pts_velo = pts_velo[:, :3]
            
        homo = np.hstack([pts_velo, np.ones((pts_velo.shape[0], 1), dtype=np.float64)])
        # X_cam = R_rect_00 * Tr_velo_to_cam * X_velo
        pts_cam = (self.R_rect_00 @ (self.Tr_velo_to_cam @ homo.T)).T[:, :3]
        return pts_cam

    def project_rect_cam_to_image2(self, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Projects 3D points in Rectified Camera coordinates to Camera 2 pixel coordinates.
        Input: (N, 3) [X, Y, Z]
        Output: u (N,), v (N,), z (N,)
        """
        homo = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float64)])
        pts_2d = (self.P2 @ homo.T).T
        z = pts_2d[:, 2]
        # Avoid division by zero
        safe_z = np.where(np.abs(z) > 1e-4, z, 1e-4)
        u = pts_2d[:, 0] / safe_z
        v = pts_2d[:, 1] / safe_z
        return u, v, z

    def project_velo_to_image2(self, pts_velo: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full projection pipeline from Velodyne points to Camera 2 pixel coordinates.
        Returns: (u, v, z, valid_mask)
        """
        # Exclude points behind Velodyne sensor
        front_velo = pts_velo[:, 0] > 0.0
        pts_velo_front = pts_velo[front_velo]

        pts_cam = self.project_velo_to_rect_cam(pts_velo_front)
        u, v, z = self.project_rect_cam_to_image2(pts_cam)

        valid = (
            (z > 0.1)
            & (u >= 0)
            & (u < self.image_width)
            & (v >= 0)
            & (v < self.image_height)
        )
        return u[valid], v[valid], z[valid], pts_cam[valid]

    def get_summary_dict(self) -> Dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "baseline_m": self.baseline,
            "fB_px_m": self.fB,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "P2_flat": self.P2.flatten().tolist(),
            "P3_flat": self.P3.flatten().tolist()
        }
