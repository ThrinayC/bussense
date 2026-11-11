#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_demo_v2_visual_integrity.py
======================================
Automated regression tests ensuring:
1. All Demo v2 and sub-reel video deliverables exist and are non-empty.
2. Sign pipeline produces dynamic bboxes and passes stale bbox test (zero static repeated boxes).
3. Missing signs produce NO fake bounding boxes.
4. KITTI vehicle pipeline produces dynamic non-static bounding boxes.
5. All QC reports (mega_demo_visual_qc.json, sign_pipeline_debug.json, etc.) pass.
"""

import json
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def test_demo_v2_video_artifacts_exist():
    videos_dir = ROOT / "outputs" / "videos"
    required_videos = [
        "bussense_mega_demo_v2.mp4",
        "bussense_kitti_verified.mp4",
        "bussense_waterlogging_verified.mp4",
        "bussense_signs_verified.mp4",
        "bussense_zebra_verified.mp4",
        "bussense_divider_verified.mp4",
        "sign_detector_raw_debug.mp4"
    ]
    for vname in required_videos:
        vpath = videos_dir / vname
        assert vpath.exists(), f"Missing required video artifact: {vname}"
        assert vpath.stat().st_size > 1024 * 1024, f"Video {vname} is too small ({vpath.stat().st_size} bytes)"

def test_sign_pipeline_anti_stale_qc():
    report_path = ROOT / "outputs" / "reports" / "sign_pipeline_debug.json"
    assert report_path.exists(), "Missing sign_pipeline_debug.json"
    with open(report_path) as f:
        data = json.load(f)
    assert data["status"] == "PASS"
    assert data["stale_bbox_test_passed"] is True
    assert data["repeat_ratio_percent"] < 5.0, f"Repeat ratio {data['repeat_ratio_percent']}% too high!"
    assert data["blank_frame_zero_box_verified"] is True
    assert data["detection_source_verified"] == "MODEL"

def test_kitti_visual_regression_qc():
    report_path = ROOT / "outputs" / "reports" / "kitti_visual_regression.json"
    assert report_path.exists(), "Missing kitti_visual_regression.json"
    with open(report_path) as f:
        data = json.load(f)
    assert data["status"] == "PASS"
    assert data["static_ratio_percent"] < 1.0, f"KITTI static box ratio {data['static_ratio_percent']}% too high!"
    assert data["total_frames_rendered"] >= 300

def test_mega_demo_visual_qc_report():
    report_path = ROOT / "outputs" / "reports" / "mega_demo_visual_qc.json"
    assert report_path.exists(), "Missing mega_demo_visual_qc.json"
    with open(report_path) as f:
        data = json.load(f)
    assert data["total_frames_rendered"] >= 3300
    assert data["duration_seconds"] >= 330.0
    assert data["anti_stale_audit"] == "PASSED (0 static bounding boxes)"
    assert data["synthetic_road_cartoon_audit"] == "PASSED (0 cartoon roads used)"
    assert "PASSED" in data["hardcoded_constants_audit"]
    assert len(data["sections"]) == 14

def test_visual_truth_and_constants_audits():
    audit_path = ROOT / "outputs" / "reports" / "visual_truth_audit.json"
    assert audit_path.exists(), "Missing visual_truth_audit.json"
    with open(audit_path) as f:
        vta = json.load(f)
    assert vta["status"] == "PASSED_ALL_INVARIANTS"
    assert vta["domains"]["waterlogging"]["status"] == "PASSED"
    assert vta["domains"]["zebra_crossing"]["status"] == "PASSED"
    assert vta["domains"]["road_divider"]["status"] == "PASSED"
    assert vta["domains"]["traffic_signs"]["status"] == "PASSED"

    constants_path = ROOT / "outputs" / "reports" / "hardcoded_constants_audit.json"
    assert constants_path.exists(), "Missing hardcoded_constants_audit.json"
    with open(constants_path) as f:
        hca = json.load(f)
    assert hca["status"] == "PASS"
    assert hca["total_violations_found"] == 0

def test_source_catalog_exists():
    cat_path = ROOT / "outputs" / "reports" / "demo_source_catalog.json"
    assert cat_path.exists(), "Missing demo_source_catalog.json"
    with open(cat_path) as f:
        data = json.load(f)
    assert len(data["video_sources"]) >= 5
    assert len(data["image_datasets"]) >= 4
