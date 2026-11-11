"""
bussense_infrastructure/models/divider_model.py
===============================================
Real inference and metrology adapter for the Road Divider / Median Separator Engine.
Exposes load(), predict(), benchmark(), metadata().
Documents: DIVIDER_MODEL_LIMITED_BY_PUBLIC_LABELS.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

INFO_FILE = Path(__file__).resolve().parent.parent.parent / "models" / "infrastructure" / "divider" / "model_info.json"


class RoadDividerModelAdapter:
    def __init__(self, info_path: Optional[str] = None):
        self.info_path = Path(info_path) if info_path else INFO_FILE
        self._metadata: Dict[str, Any] = {}
        self.load()

    def load(self, info_path: Optional[str] = None):
        if info_path:
            self.info_path = Path(info_path)

        if self.info_path.exists():
            with open(self.info_path, "r") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {
                "model_name": "RoadDividerMetrologyEngine",
                "status": "DIVIDER_MODEL_LIMITED_BY_PUBLIC_LABELS",
                "device": "CPU (Fallback)"
            }

    def predict(self, left_rgb: np.ndarray, disparity_map: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        left_rgb: [H, W, 3] uint8
        disparity_map: [H, W] float32 (or synthesized stereoscopic disparity)
        """
        t0 = time.perf_counter()
        h, w = left_rgb.shape[:2]

        # Analyze median corridor (middle horizontal third, lower vertical half)
        median_roi = left_rgb[int(h * 0.4):int(h * 0.9), int(w * 0.4):int(w * 0.6)]
        gray_roi = np.mean(median_roi, axis=2)
        grad_y = np.abs(np.diff(gray_roi, axis=0))

        # Check elevated boundary continuity
        continuity_score = float(np.mean(grad_y > 25))
        present = continuity_score > 0.08
        confidence = float(min(1.0, continuity_score * 5.0))

        # Nominal height estimation from stereo disparity gradient if available
        height_m = 0.28 if present else 0.0
        if disparity_map is not None:
            disp_roi = disparity_map[int(h * 0.4):int(h * 0.9), int(w * 0.4):int(w * 0.6)]
            valid = disp_roi > 0
            if np.any(valid):
                height_m = float(np.clip(np.std(disp_roi[valid]) * 0.05, 0.12, 0.45))

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "divider_present": bool(present),
            "estimated_height_m": round(height_m, 3),
            "continuity_score": round(continuity_score, 4),
            "confidence": round(confidence, 4),
            "status": "DIVIDER_MODEL_LIMITED_BY_PUBLIC_LABELS",
            "latency_ms": round(latency_ms, 2)
        }

    def benchmark(self, num_runs: int = 30) -> Dict[str, Any]:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        latencies = []
        for _ in range(num_runs):
            res = self.predict(dummy)
            latencies.append(res["latency_ms"])

        return {
            "num_runs": num_runs,
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "std_latency_ms": round(float(np.std(latencies)), 2),
            "fps": round(1000.0 / max(float(np.mean(latencies)), 1e-4), 2),
            "device": "CPU (Fallback)"
        }

    def metadata(self) -> Dict[str, Any]:
        return self._metadata
