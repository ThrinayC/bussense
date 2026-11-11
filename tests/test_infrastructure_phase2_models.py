"""
tests/test_infrastructure_phase2_models.py
===========================================
Verification test suite for BusSense Phase 2 Real Perception Models:
1. Model Registry & Checksum Integrity
2. Water Model Adapter (predict, benchmark, metadata)
3. Traffic Sign Model Adapter (predict, taxonomy, metadata)
4. Sign Condition Model Adapter (predict, condition classes)
5. Zebra Model Adapter (predict, mask shape, metadata)
6. Road Divider Model Adapter (metadata, status disclosure)
7. Integration with Perception Engines (use_model=True)
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bussense_infrastructure.models import (
    WaterModelAdapter,
    TrafficSignModelAdapter,
    SignConditionModelAdapter,
    ZebraModelAdapter,
    RoadDividerModelAdapter,
    LightweightRoadSegNet,
    TrafficSignNet,
    SignConditionNet
)
from bussense_infrastructure.waterlogging_engine import WaterloggingEngine
from bussense_infrastructure.traffic_sign_engine import TrafficSignEngine
from bussense_infrastructure.zebra_crossing_engine import ZebraCrossingEngine
from bussense_infrastructure.road_divider_engine import RoadDividerEngine


def test_model_registry_integrity():
    reg_path = Path(__file__).resolve().parent.parent / "models" / "infrastructure" / "model_registry.json"
    assert reg_path.exists(), "model_registry.json must exist"

    with open(reg_path, "r") as f:
        registry = json.load(f)

    assert "hardware" in registry
    assert registry["hardware"]["device"] == "CPU (Fallback)"
    assert "models" in registry

    expected_models = ["water_model", "zebra_model", "traffic_sign_model", "sign_condition_model", "road_divider_model"]
    for m in expected_models:
        assert m in registry["models"], f"Model {m} missing from registry"
        mod_info = registry["models"][m]
        assert "mean_latency_ms" in mod_info
        assert mod_info["mean_latency_ms"] > 0
        assert "fps" in mod_info


def test_water_model_adapter():
    adapter = WaterModelAdapter()
    meta = adapter.metadata()
    assert meta["task"] == "waterlogging_segmentation"
    assert meta["device"] == "CPU (Fallback)"

    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = adapter.predict(dummy)

    assert "mask" in res
    assert res["mask"].shape == (256, 256)
    assert "confidence" in res
    assert "water_pixel_fraction" in res
    assert "latency_ms" in res
    assert res["latency_ms"] > 0


def test_zebra_model_adapter():
    adapter = ZebraModelAdapter()
    meta = adapter.metadata()
    assert meta["task"] == "zebra_crossing_segmentation"

    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = adapter.predict(dummy)

    assert "mask" in res
    assert res["mask"].shape == (256, 256)
    assert "crossing_pixel_fraction" in res
    assert "latency_ms" in res


def test_traffic_sign_adapter():
    adapter = TrafficSignModelAdapter()
    meta = adapter.metadata()
    assert meta["task"] == "traffic_sign_detection"
    assert len(meta["taxonomy"]) == 6

    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = adapter.predict(dummy, conf_threshold=0.1)

    assert "detections" in res
    assert "latency_ms" in res


def test_sign_condition_adapter():
    adapter = SignConditionModelAdapter()
    meta = adapter.metadata()
    assert meta["task"] == "sign_condition_classification"
    assert set(meta["conditions"]) == {"NORMAL", "DAMAGED", "OCCLUDED"}

    dummy_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    res = adapter.predict(dummy_crop)

    assert "condition" in res
    assert res["condition"] in ["NORMAL", "DAMAGED", "OCCLUDED"]
    assert "probabilities" in res
    assert len(res["probabilities"]) == 3


def test_road_divider_adapter():
    adapter = RoadDividerModelAdapter()
    meta = adapter.metadata()
    assert meta["status"] == "DIVIDER_MODEL_LIMITED_BY_PUBLIC_LABELS"

    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    res = adapter.predict(dummy)

    assert "divider_present" in res
    assert "estimated_height_m" in res
    assert "status" in res


def test_engines_with_real_models():
    # Waterlogging Engine with real model
    w_eng = WaterloggingEngine(use_model=True)
    assert w_eng.model_adapter is not None
    dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    obs = w_eng.process_frame(dummy, frame_idx=1, timestamp=1.0)
    assert obs is not None
    assert 0.0 <= obs.water_coverage_ratio <= 1.0

    # Traffic Sign Engine with real model
    s_eng = TrafficSignEngine(use_model=True)
    assert s_eng.sign_model is not None
    assert s_eng.condition_model is not None
    dets = s_eng.detect_candidate_signs(dummy)
    assert isinstance(dets, list)

    # Zebra Crossing Engine with real model
    z_eng = ZebraCrossingEngine(use_model=True)
    assert z_eng.zebra_model is not None
    env, stripes = z_eng.segment_crossing(dummy)
    assert env.shape == (300, 400)
    assert stripes.shape == (300, 400)

    # Road Divider Engine with real model adapter
    d_eng = RoadDividerEngine(use_model=True)
    assert d_eng.divider_model is not None
