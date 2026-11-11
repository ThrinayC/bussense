"""
StereoEngine Abstract Base Class
Unified interface for all stereo depth estimation backends (SGBM, IGEV).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np


class StereoEngine(ABC):
    """
    Common contract for stereo matching engines.
    Ensures that downstream perception (vehicles, VRUs, kinematics, evidence)
    is completely decoupled from whether SGBM, IGEV, or another backend computes disparity.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_disparity(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Computes disparity between rectified left and right images.
        Input:
            left_img: (H, W, 3) or (H, W) uint8
            right_img: (H, W, 3) or (H, W) uint8
        Returns:
            disparity: (H, W) float32 in pixels. Invalid pixels should be <= 0.
            confidence: (H, W) float32 in [0, 1] or None if backend doesn't support.
        """
        pass

    def disparity_confidence(
        self, disparity: np.ndarray, raw_disp16: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Derives an empirical confidence map [0.0, 1.0] from disparity quality cues.
        """
        conf = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0.5
        conf[valid] = 1.0
        return conf

    def disparity_to_depth(
        self, disparity: np.ndarray, fB: float, min_depth: float = 0.5, max_depth: float = 120.0
    ) -> np.ndarray:
        """
        Converts disparity to metric depth: Z = (fx * baseline) / disparity.
        Invalid disparities (<= 0) are mapped to 0.0 or NaN.
        """
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0.1
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_depth = fB / np.where(valid, disparity, np.nan)
            valid_range = valid & (raw_depth >= min_depth) & (raw_depth <= max_depth)
            depth[valid_range] = raw_depth[valid_range]
        return depth

    def depth_to_xyz(
        self, depth: np.ndarray, fx: float, fy: float, cx: float, cy: float
    ) -> np.ndarray:
        """
        Unprojects a 2D depth map into a 3D point cloud in camera coordinates.
        Returns (H, W, 3) where channels are (X, Y, Z). Invalid depths have Z = 0.
        """
        H, W = depth.shape
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        
        valid = depth > 0.1
        X = np.zeros_like(depth)
        Y = np.zeros_like(depth)
        Z = np.copy(depth)

        X[valid] = (u_grid[valid] - cx) * depth[valid] / fx
        Y[valid] = (v_grid[valid] - cy) * depth[valid] / fy

        xyz = np.stack([X, Y, Z], axis=-1)
        return xyz

    @abstractmethod
    def get_backend_info(self) -> Dict:
        """
        Returns metadata: backend name, device (CPU/GPU), runtime statistics, configuration.
        """
        pass
