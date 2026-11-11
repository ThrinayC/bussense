#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Evidence-Based Road Quality Engine
=============================================
Constructs a measurable physical road condition vector:
- Pothole count, area ratio, depth statistics
- Crack length density (m/m²) and crack surface coverage ratio
- Stereo Surface Irregularity / Plane Residual RMS (cm) [NOT IRI]
- Observable road area (m²)
- Measurement confidence (%)
- Documented BusSense provisional category

Strict guardrail:
Returns "ROAD QUALITY: UNRELIABLE" if road plane or stereo geometry is insufficient.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


from collections import deque

OPERATIONAL_LEVELS = {
    "NORMAL": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "ALARMING": 3
}

# Configurable BusSense Operational Thresholds (Physical Evidence Hierarchy)
BUSSENSE_OPERATIONAL_THRESHOLDS = {
    # Pothole primary criteria
    "pothole_high_sev_depth_cm": 5.0,
    "pothole_alarming_depth_cm": 7.0,
    "pothole_alarming_area_m2": 0.40,
    "pothole_alarming_area_ratio": 0.03,
    "pothole_high_depth_cm": 2.5,
    "pothole_high_area_m2": 0.10,

    # Crack primary criteria
    "crack_alarming_density_m_m2": 0.25,
    "crack_high_density_m_m2": 0.06,
    "crack_medium_density_m_m2": 0.015,
    "crack_high_count": 3,

    # Stereo surface irregularity (secondary evidence only)
    "surface_rms_alarming_cm": 12.0,
    "surface_rms_high_cm": 8.0,
    "surface_rms_medium_cm": 5.0,
}

@dataclass
class RoadQualityMetrics:
    status: str                                    # "OK" or "ROAD QUALITY: UNRELIABLE"
    condition_category: str                        # "NORMAL", "MEDIUM", "HIGH", "ALARMING", or "UNRELIABLE"
    classification_basis: str                      # "BusSense Evidence-Based Operational Road Quality v1.1"
    confidence_percent: float                      # 0.0 - 100.0%
    observable_road_area_m2: Optional[float]
    pothole_count: int
    pothole_area_ratio: float                      # Fraction of observable road
    pothole_max_depth_cm: Optional[float]
    pothole_mean_depth_cm: Optional[float]
    high_severity_potholes: int                    # depth >= 5.0cm
    crack_count: int
    crack_length_m: float
    crack_length_density_m_m2: Optional[float]
    crack_area_ratio: float
    stereo_surface_irregularity_rms_cm: Optional[float]  # Plane Residual RMS [NOT IRI]
    evidence_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def operational_level(self) -> str:
        return self.condition_category

    @property
    def instant_level(self) -> str:
        return self.evidence_summary.get("instant_level", self.condition_category)

    @property
    def smoothed_level(self) -> str:
        return self.condition_category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "condition_category": self.condition_category,
            "classification_basis": self.classification_basis,
            "confidence_percent": round(self.confidence_percent, 1),
            "observable_road_area_m2": round(self.observable_road_area_m2, 2) if self.observable_road_area_m2 else None,
            "pothole_count": self.pothole_count,
            "pothole_area_ratio": round(self.pothole_area_ratio, 6),
            "pothole_max_depth_cm": round(self.pothole_max_depth_cm, 1) if self.pothole_max_depth_cm is not None else None,
            "pothole_mean_depth_cm": round(self.pothole_mean_depth_cm, 1) if self.pothole_mean_depth_cm is not None else None,
            "high_severity_potholes": self.high_severity_potholes,
            "crack_count": self.crack_count,
            "crack_length_m": round(self.crack_length_m, 2),
            "crack_length_density_m_m2": round(self.crack_length_density_m_m2, 4) if self.crack_length_density_m_m2 is not None else None,
            "crack_area_ratio": round(self.crack_area_ratio, 6),
            "stereo_surface_irregularity_rms_cm": round(self.stereo_surface_irregularity_rms_cm, 2) if self.stereo_surface_irregularity_rms_cm is not None else None,
            "evidence_summary": self.evidence_summary
        }


class RoadQualityEngine:
    """Evaluates physical road condition from stereo geometry, potholes, and cracks."""

    LEVELS = ["NORMAL", "MEDIUM", "HIGH", "ALARMING"]

    def __init__(self, high_severity_depth_thresh_cm: float = 5.0, history_len: int = 8):
        self.high_severity_depth_thresh_cm = high_severity_depth_thresh_cm
        self.history = deque(maxlen=history_len)
        self.current_level_idx = 0  # 0: NORMAL

    def reset(self):
        """Resets temporal smoothing history."""
        self.history.clear()
        self.current_level_idx = 0

    def _compute_plane_residual_rms(
        self,
        depth_m: np.ndarray,
        plane: np.ndarray,
        calib: Dict[str, float]
    ) -> Optional[float]:
        """
        Computes the Stereo Surface Irregularity / Plane Residual RMS in cm.
        Samples depth points on the lower road surface and computes orthogonal
        distance to the fitted road plane.
        """
        if depth_m is None or plane is None:
            return None

        H, W = depth_m.shape[:2]
        road_y0 = int(H * 0.65)
        roi_depth = depth_m[road_y0:H, :]
        valid = (roi_depth >= 1.0) & (roi_depth <= 25.0) & np.isfinite(roi_depth)

        if np.count_nonzero(valid) < 80:
            return None

        ys, xs = np.where(valid)
        if len(ys) > 1500:
            step = len(ys) / 1500.0
            idx = np.array([int(i * step) for i in range(1500)], dtype=np.int32)
            ys = ys[idx]
            xs = xs[idx]

        zs = roi_depth[ys, xs]
        xs_3d = (xs - calib["cx"]) * zs / calib["fx"]
        ys_3d = (ys + road_y0 - calib["cy"]) * zs / calib["fy"]

        # Orthogonal distance: |a*X + b*Y + c*Z + d| / ||n||
        a, b, c, d = plane
        norm = np.sqrt(a * a + b * b + c * c)
        if norm < 1e-5:
            return None

        distances_m = np.abs(a * xs_3d + b * ys_3d + c * zs + d) / norm
        # Surface metrology: reject non-pavement obstacles (vehicles, curbs, signposts > 15cm from plane)
        pavement_dists = distances_m[distances_m <= 0.15]
        if len(pavement_dists) < 50:
            return None
        rms_m = float(np.sqrt(np.mean(pavement_dists ** 2)))
        return rms_m * 100.0  # Convert to cm

    def evaluate_road_quality(
        self,
        potholes: List[Dict[str, Any]],
        cracks: List[Dict[str, Any]],
        road_plane_meta: Optional[Dict[str, Any]],
        depth_m: Optional[np.ndarray],
        calib: Dict[str, float],
        observable_road_area_m2: Optional[float] = None,
        stereo_quality: Optional[Dict[str, Any]] = None
    ) -> RoadQualityMetrics:
        """Evaluates full physical condition vector for the current road segment."""
        # Check guardrails
        has_valid_plane = (road_plane_meta is not None and road_plane_meta.get("is_valid", False))
        has_valid_disp = (stereo_quality is not None and stereo_quality.get("valid_disp_fraction", 0.0) >= 0.10)
        has_valid_area = (observable_road_area_m2 is not None and observable_road_area_m2 >= 3.0)

        if not has_valid_plane or not has_valid_disp or not has_valid_area:
            return RoadQualityMetrics(
                status="ROAD CONDITION: INSUFFICIENT DATA",
                condition_category="INSUFFICIENT DATA",
                classification_basis="Insufficient valid road plane geometry or stereo coverage",
                confidence_percent=0.0,
                observable_road_area_m2=observable_road_area_m2,
                pothole_count=len(potholes),
                pothole_area_ratio=0.0,
                pothole_max_depth_cm=None,
                pothole_mean_depth_cm=None,
                high_severity_potholes=0,
                crack_count=len(cracks),
                crack_length_m=0.0,
                crack_length_density_m_m2=None,
                crack_area_ratio=0.0,
                stereo_surface_irregularity_rms_cm=None,
                evidence_summary={"failure_reason": "Guardrail triggered: Plane or stereo unverified"}
            )

        # 1. Pothole distress statistics
        p_count = len(potholes)
        p_depths = [p.get("depth_p95_cm") for p in potholes if p.get("depth_p95_cm") is not None]
        p_areas = [p.get("area_m2") for p in potholes if p.get("area_m2") is not None]

        p_max_depth = float(max(p_depths)) if p_depths else 0.0
        p_mean_depth = float(np.mean(p_depths)) if p_depths else 0.0
        p_total_area = float(sum(p_areas)) if p_areas else 0.0
        p_area_ratio = float(p_total_area / observable_road_area_m2) if observable_road_area_m2 else 0.0
        high_sev_count = int(sum(1 for d in p_depths if d >= self.high_severity_depth_thresh_cm))

        # 2. Crack distress statistics
        c_count = len(cracks)
        c_lengths = [c.get("length_m", 0.0) or 0.0 for c in cracks]
        c_areas = [c.get("coverage_area_m2", 0.0) or 0.0 for c in cracks]

        c_total_len = float(sum(c_lengths)) if c_lengths else 0.0
        c_total_area = float(sum(c_areas)) if c_areas else 0.0
        c_density = float(c_total_len / observable_road_area_m2) if observable_road_area_m2 else 0.0
        c_area_ratio = float(c_total_area / observable_road_area_m2) if observable_road_area_m2 else 0.0

        # 3. Stereo Surface Irregularity (Secondary Evidence)
        plane_eq = np.array(road_plane_meta["equation"], dtype=np.float32)
        surface_rms_cm = self._compute_plane_residual_rms(depth_m, plane_eq, calib)

        # 4. Measurement Confidence
        inlier_ratio = road_plane_meta.get("inlier_ratio", 0.5)
        disp_frac = stereo_quality.get("valid_disp_fraction", 0.5)
        confidence = float(np.clip((inlier_ratio * 0.6 + disp_frac * 0.4) * 100.0, 15.0, 98.0))

        # 5. Dynamic Instantaneous Road Condition Classification
        # Operational States: NORMAL, MEDIUM, HIGH, ALARMING
        T = BUSSENSE_OPERATIONAL_THRESHOLDS
        if (
            high_sev_count >= 1 or
            p_max_depth >= T["pothole_alarming_depth_cm"] or
            p_total_area >= T["pothole_alarming_area_m2"] or
            p_area_ratio >= T["pothole_alarming_area_ratio"] or
            c_density >= T["crack_alarming_density_m_m2"] or
            (p_count >= 2 and p_max_depth >= 4.0) or
            (surface_rms_cm is not None and surface_rms_cm >= T["surface_rms_alarming_cm"] and (p_count > 0 or c_count > 0))
        ):
            instant_level = 3  # ALARMING
        elif (
            (p_count >= 1 and (p_max_depth >= T["pothole_high_depth_cm"] or p_total_area >= T["pothole_high_area_m2"])) or
            p_count >= 2 or
            c_density >= T["crack_high_density_m_m2"] or
            c_count >= T["crack_high_count"] or
            (surface_rms_cm is not None and surface_rms_cm >= T["surface_rms_high_cm"])
        ):
            instant_level = 2  # HIGH
        elif (
            p_count >= 1 or
            c_count >= 1 or
            c_density >= T["crack_medium_density_m_m2"] or
            (surface_rms_cm is not None and surface_rms_cm >= T["surface_rms_medium_cm"])
        ):
            instant_level = 1  # MEDIUM
        else:
            instant_level = 0  # NORMAL

        # 6. Temporal Smoothing & Hysteresis
        self.history.append(instant_level)

        # Transition logic with hysteresis:
        # Upward transition: fast response (requires 2 consecutive frames or instant if alarming with depth >= 8cm)
        # Downward transition: conservative smoothing (requires >= 4 consecutive lower frames)
        if len(self.history) >= 2:
            last_two = list(self.history)[-2:]
            if all(lvl > self.current_level_idx for lvl in last_two):
                self.current_level_idx = min(last_two)
            elif instant_level == 3 and p_max_depth >= 8.0:
                self.current_level_idx = 3  # Immediate alert for severe deep potholes
            elif len(self.history) >= 4 and all(lvl < self.current_level_idx for lvl in list(self.history)[-4:]):
                self.current_level_idx = max(list(self.history)[-4:])
        else:
            # First frame: require 2 consecutive frames for upgrade unless immediate emergency alarm (>= 8cm)
            if instant_level == 3 and p_max_depth >= 8.0:
                self.current_level_idx = 3
            else:
                self.current_level_idx = 0  # Start at NORMAL until 2nd frame confirms upgrade

        final_category = self.LEVELS[self.current_level_idx]

        evidence = {
            "instant_level": self.LEVELS[instant_level],
            "smoothed_level": final_category,
            "pothole_distress": f"{p_count} detected (max depth {p_max_depth:.1f} cm, total area {p_total_area:.3f} m²)",
            "crack_distress": f"{c_count} detected ({c_total_len:.1f} m, {c_density:.3f} m/m²)",
            "surface_roughness": f"Residual RMS: {surface_rms_cm:.2f} cm" if surface_rms_cm else "N/A",
            "plane_inliers": f"{inlier_ratio*100:.1f}%",
            "observable_area": f"{observable_road_area_m2:.1f} m²"
        }

        return RoadQualityMetrics(
            status="OK",
            condition_category=final_category,
            classification_basis="BusSense Evidence-Based Operational Road Quality v1.2",
            confidence_percent=confidence,
            observable_road_area_m2=observable_road_area_m2,
            pothole_count=p_count,
            pothole_area_ratio=p_area_ratio,
            pothole_max_depth_cm=p_max_depth if p_count > 0 else None,
            pothole_mean_depth_cm=p_mean_depth if p_count > 0 else None,
            high_severity_potholes=high_sev_count,
            crack_count=c_count,
            crack_length_m=c_total_len,
            crack_length_density_m_m2=c_density,
            crack_area_ratio=c_area_ratio,
            stereo_surface_irregularity_rms_cm=surface_rms_cm,
            evidence_summary=evidence
        )

    assess = evaluate_road_quality
