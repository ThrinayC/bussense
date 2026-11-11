"""
OpenCV Semi-Global Block Matching (SGBM) Stereo Engine
Implements the StereoEngine contract as the declared CPU/baseline stereo processor.
"""

import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from scripts.stereo_engine import StereoEngine


class SGBMStereoEngine(StereoEngine):
    """
    OpenCV StereoSGBM implementation of StereoEngine.
    """

    def __init__(
        self,
        minDisparity: int = 0,
        numDisparities: int = 128,
        blockSize: int = 5,
        P1: Optional[int] = None,
        P2: Optional[int] = None,
        disp12MaxDiff: int = 1,
        uniquenessRatio: int = 10,
        speckleWindowSize: int = 100,
        speckleRange: int = 32,
        mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    ):
        super().__init__(name="OpenCV_SGBM")
        self.minDisparity = minDisparity
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        self.disp12MaxDiff = disp12MaxDiff
        self.uniquenessRatio = uniquenessRatio
        self.speckleWindowSize = speckleWindowSize
        self.speckleRange = speckleRange
        self.mode = mode

        # Standard SGBM energy parameters scaled by channel count (3) and window area
        channels = 3
        if P1 is None:
            self.P1 = 8 * channels * (self.blockSize**2)
        else:
            self.P1 = P1

        if P2 is None:
            self.P2 = 32 * channels * (self.blockSize**2)
        else:
            self.P2 = P2

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=self.minDisparity,
            numDisparities=self.numDisparities,
            blockSize=self.blockSize,
            P1=self.P1,
            P2=self.P2,
            disp12MaxDiff=self.disp12MaxDiff,
            uniquenessRatio=self.uniquenessRatio,
            speckleWindowSize=self.speckleWindowSize,
            speckleRange=self.speckleRange,
            mode=self.mode,
        )

        self.last_latency_ms: float = 0.0
        self.call_count: int = 0
        self.total_latency_ms: float = 0.0

    def compute_disparity(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Computes disparity and empirical confidence map.
        OpenCV SGBM outputs 16-bit fixed point disparity with 4 fractional bits.
        """
        t0 = time.perf_counter()

        # OpenCV SGBM expects BGR or Grayscale
        disp16 = self.matcher.compute(left_img, right_img)
        t1 = time.perf_counter()

        self.last_latency_ms = (t1 - t0) * 1000.0
        self.total_latency_ms += self.last_latency_ms
        self.call_count += 1

        # Convert fixed point to floating point pixels
        disp = disp16.astype(np.float32) / 16.0

        # Disparity confidence estimation:
        # SGBM sets invalid disparities to (minDisparity - 1) or values < minDisparity
        valid_mask = disp > (self.minDisparity + 0.1)

        # Build confidence map based on local texture and disparity validity
        confidence = np.zeros_like(disp, dtype=np.float32)
        confidence[valid_mask] = 0.85  # baseline confidence for valid SGBM match

        # Penalize low texture / flat regions where disparity might be speckle-prone
        if len(left_img.shape) == 3:
            gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = left_img
            
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        high_texture = grad_mag > 15.0

        confidence[valid_mask & high_texture] = 1.0
        confidence[~valid_mask] = 0.0

        return disp, confidence

    def get_backend_info(self) -> Dict:
        avg_latency = (
            self.total_latency_ms / max(1, self.call_count) if self.call_count > 0 else 0.0
        )
        return {
            "backend": self.name,
            "device": "CPU",
            "numDisparities": self.numDisparities,
            "blockSize": self.blockSize,
            "P1": self.P1,
            "P2": self.P2,
            "uniquenessRatio": self.uniquenessRatio,
            "speckleWindowSize": self.speckleWindowSize,
            "mode": self.mode,
            "last_latency_ms": round(self.last_latency_ms, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "call_count": self.call_count,
        }
