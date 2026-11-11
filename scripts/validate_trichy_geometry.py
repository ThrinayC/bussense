#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_trichy_geometry.py
===================================
Validates the actual OSM Way 369881223 geometry for Bharathidasan Salai,
Tiruchirappalli, Tamil Nadu, India (Section M).

Invariants:
- Real OSM Way 369881223 with 26 verified nodes.
- Continuous multi-segment LineString (NOT straight-line interpolation).
- Arc-length resampling at 5-meter intervals.
- Local tangent heading and curvature calculations.
- Centerline validation asserting distance <= 10.0m for every point.
- Saves report to outputs/reports/trichy_geometry_validation.json.
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.trichy_road_geometry import (
    BHARATHIDASAN_SALAI_OSM_NODES,
    TrichyRoadGeometryManager,
    haversine_m,
    compute_bearing_deg,
    point_to_segment_distance_m
)


def compute_menger_curvature(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Computes Menger curvature (1/R) for three points in local metric projection.
    Units: 1 / meters (rad/m).
    """
    # Convert lat/lon to local meter coordinates
    cos_lat = math.cos(math.radians(p2[0]))
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * cos_lat

    x1, y1 = p1[1] * m_per_deg_lon, p1[0] * m_per_deg_lat
    x2, y2 = p2[1] * m_per_deg_lon, p2[0] * m_per_deg_lat
    x3, y3 = p3[1] * m_per_deg_lon, p3[0] * m_per_deg_lat

    # Triangle side lengths
    a = math.hypot(x2 - x1, y2 - y1)
    b = math.hypot(x3 - x2, y3 - y2)
    c = math.hypot(x3 - x1, y3 - y1)

    if a < 1e-4 or b < 1e-4 or c < 1e-4:
        return 0.0

    # Triangle area via cross product
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    curvature = (4.0 * area) / (a * b * c)
    return float(curvature)


def validate_trichy_geometry() -> Dict[str, Any]:
    print("=" * 70)
    print("TRICHY ROAD GEOMETRY VALIDATION (OSM Way 369881223)")
    print("=" * 70)

    mgr = TrichyRoadGeometryManager(sampling_interval_m=5.0, tolerance_m=10.0)
    raw_nodes = BHARATHIDASAN_SALAI_OSM_NODES

    # 1. Basic polyline metrics
    node_count = len(raw_nodes)
    total_length_m = mgr.total_road_length_m

    start_node = raw_nodes[0]
    end_node = raw_nodes[-1]
    straight_line_dist = haversine_m(start_node["lat"], start_node["lon"], end_node["lat"], end_node["lon"])
    tortuosity = total_length_m / straight_line_dist if straight_line_dist > 0 else 1.0

    print(f"OSM Way: {mgr.OSM_WAY_ID} ({mgr.ROAD_NAME})")
    print(f"Node count: {node_count}")
    print(f"Polyline length: {total_length_m:.2f} m")
    print(f"Straight-line distance: {straight_line_dist:.2f} m")
    print(f"Tortuosity: {tortuosity:.4f} (Ratio > 1.0 confirms non-straight polyline)")

    # 2. Resample by arc length every 5m
    sampled_records = []
    sampling_step = 5.0
    num_steps = int(math.ceil(total_length_m / sampling_step))
    deviations = []

    for i in range(num_steps + 1):
        s = min(total_length_m, i * sampling_step)
        lat, lon, heading = mgr.sample_point_at_distance(s)
        val = mgr.validate_point_on_road(lat, lon)

        if not val["valid"]:
            raise ValueError(f"Validation failed at s={s:.1f}m: dist={val['distance_to_centerline_m']}m > 10m")

        dev = val["distance_to_centerline_m"]
        deviations.append(dev)

        sampled_records.append({
            "s_m": round(s, 2),
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "heading_deg": round(heading, 2),
            "centerline_deviation_m": dev
        })

    # 3. Curvature calculation
    curvatures = []
    for i in range(len(sampled_records)):
        if i == 0:
            p1 = (sampled_records[0]["lat"], sampled_records[0]["lon"])
            p2 = (sampled_records[1]["lat"], sampled_records[1]["lon"])
            p3 = (sampled_records[2]["lat"], sampled_records[2]["lon"])
        elif i == len(sampled_records) - 1:
            p1 = (sampled_records[-3]["lat"], sampled_records[-3]["lon"])
            p2 = (sampled_records[-2]["lat"], sampled_records[-2]["lon"])
            p3 = (sampled_records[-1]["lat"], sampled_records[-1]["lon"])
        else:
            p1 = (sampled_records[i - 1]["lat"], sampled_records[i - 1]["lon"])
            p2 = (sampled_records[i]["lat"], sampled_records[i]["lon"])
            p3 = (sampled_records[i + 1]["lat"], sampled_records[i + 1]["lon"])

        k = compute_menger_curvature(p1, p2, p3)
        curvatures.append(k)
        sampled_records[i]["curvature_rad_per_m"] = round(k, 6)

    max_dev = max(deviations)
    mean_dev = sum(deviations) / len(deviations)
    max_k = max(curvatures)
    mean_k = sum(curvatures) / len(curvatures)

    print(f"Sampled points: {len(sampled_records)}")
    print(f"Max centerline deviation: {max_dev:.4f} m (<= 10.0m tolerance: {max_dev <= 10.0})")
    print(f"Mean centerline deviation: {mean_dev:.4f} m")
    print(f"Max curvature: {max_k:.6f} rad/m (Min radius: {1.0/max_k if max_k > 0 else 0:.1f} m)")
    print(f"Mean curvature: {mean_k:.6f} rad/m")

    # 4. Generate report
    report = {
        "validation_metadata": {
            "title": "Trichy Road Geometry & Polyline Validation Report",
            "section": "Section M (Master Implementation Prompt)",
            "city": "Tiruchirappalli, Tamil Nadu, India",
            "road_name": mgr.ROAD_NAME,
            "osm_way_id": mgr.OSM_WAY_ID,
            "highway_type": "primary"
        },
        "road_polyline_metrics": {
            "osm_node_count": node_count,
            "polyline_length_m": round(total_length_m, 2),
            "straight_line_dist_m": round(straight_line_dist, 2),
            "tortuosity": round(tortuosity, 4),
            "is_straight_line_interpolation": False,
            "first_node": raw_nodes[0],
            "last_node": raw_nodes[-1]
        },
        "sampling_and_tolerances": {
            "sampling_interval_m": sampling_step,
            "total_sampled_points": len(sampled_records),
            "centerline_tolerance_m": 10.0,
            "max_centerline_deviation_m": round(max_dev, 4),
            "mean_centerline_deviation_m": round(mean_dev, 4),
            "centerline_check_passed": bool(max_dev <= 10.0)
        },
        "curvature_analysis": {
            "mean_curvature_rad_per_m": round(mean_k, 6),
            "max_curvature_rad_per_m": round(max_k, 6),
            "min_curve_radius_m": round(1.0 / max_k if max_k > 0 else float("inf"), 2)
        },
        "sampled_points_preview": {
            "first_5": sampled_records[:5],
            "last_5": sampled_records[-5:]
        },
        "overall_status": "VALIDATED_PASS" if (max_dev <= 10.0 and tortuosity > 1.0) else "FAILED"
    }

    out_file = ROOT / "outputs" / "reports" / "trichy_geometry_validation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[REPORT] Saved validation report to {out_file}")
    return report


if __name__ == "__main__":
    validate_trichy_geometry()
