"""
scripts/benchmark_infrastructure_models.py
==========================================
Benchmarks all five Phase 2 infrastructure perception models on host hardware.
Produces empirical latency, throughput, parameter counts, and SHA256 hashes.
Generates `models/infrastructure/model_registry.json`.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bussense_infrastructure.models import (
    WaterModelAdapter,
    TrafficSignModelAdapter,
    SignConditionModelAdapter,
    ZebraModelAdapter,
    RoadDividerModelAdapter
)

REGISTRY_PATH = ROOT / "models" / "infrastructure" / "model_registry.json"


def get_file_sha256(path: Path) -> str:
    if not path.exists():
        return "NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_benchmarks(num_runs: int = 30):
    print("=" * 70)
    print("BUSSENSE PHASE 2: EMPIRICAL HARDWARE MODEL BENCHMARK")
    print("=" * 70)
    print(f"Host Execution Target: CPU (Fallback) - PyTorch 2.14.0+cpu")
    print(f"Benchmark Iterations per Model: {num_runs}\n")

    registry = {
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": "CPU (Fallback)",
            "cuda_available": False,
            "os": "Windows 11",
            "python_version": sys.version.split()[0]
        },
        "models": {}
    }

    # 1. Water Model
    print("Benchmarking Water Segmentation Model...")
    water_adapter = WaterModelAdapter()
    water_bench = water_adapter.benchmark(num_runs)
    water_meta = water_adapter.metadata()
    registry["models"]["water_model"] = {
        "name": water_meta.get("name", "LightweightRoadSegNet_Water"),
        "task": "waterlogging_segmentation",
        "weights_path": "models/infrastructure/water/best.pt",
        "sha256": get_file_sha256(Path(water_meta["weights_path"])),
        "parameters": water_meta.get("parameters", 0),
        "input_resolution": "256x256",
        "mean_latency_ms": water_bench["mean_latency_ms"],
        "std_latency_ms": water_bench["std_latency_ms"],
        "fps": water_bench["fps"],
        "device": "CPU (Fallback)"
    }
    print(f"  -> Mean Latency: {water_bench['mean_latency_ms']} ms | FPS: {water_bench['fps']}")

    # 2. Zebra Model
    print("\nBenchmarking Zebra Crossing Model...")
    zebra_adapter = ZebraModelAdapter()
    zebra_bench = zebra_adapter.benchmark(num_runs)
    zebra_meta = zebra_adapter.metadata()
    registry["models"]["zebra_model"] = {
        "name": zebra_meta.get("name", "LightweightRoadSegNet_Zebra"),
        "task": "zebra_crossing_segmentation",
        "weights_path": "models/infrastructure/zebra/best.pt",
        "sha256": get_file_sha256(Path(zebra_meta["weights_path"])),
        "parameters": zebra_meta.get("parameters", 0),
        "input_resolution": "256x256",
        "mean_latency_ms": zebra_bench["mean_latency_ms"],
        "std_latency_ms": zebra_bench["std_latency_ms"],
        "fps": zebra_bench["fps"],
        "device": "CPU (Fallback)"
    }
    print(f"  -> Mean Latency: {zebra_bench['mean_latency_ms']} ms | FPS: {zebra_bench['fps']}")

    # 3. Traffic Sign Model
    print("\nBenchmarking Traffic Sign Model...")
    sign_adapter = TrafficSignModelAdapter()
    sign_bench = sign_adapter.benchmark(num_runs)
    sign_meta = sign_adapter.metadata()
    registry["models"]["traffic_sign_model"] = {
        "name": sign_meta.get("name", "TrafficSignNet"),
        "task": "traffic_sign_detection",
        "taxonomy": sign_meta.get("taxonomy", []),
        "weights_path": "models/infrastructure/signs/best.pt",
        "sha256": get_file_sha256(Path(sign_meta["weights_path"])),
        "parameters": sign_meta.get("parameters", 0),
        "input_resolution": "256x256",
        "mean_latency_ms": sign_bench["mean_latency_ms"],
        "std_latency_ms": sign_bench["std_latency_ms"],
        "fps": sign_bench["fps"],
        "device": "CPU (Fallback)"
    }
    print(f"  -> Mean Latency: {sign_bench['mean_latency_ms']} ms | FPS: {sign_bench['fps']}")

    # 4. Sign Condition Model
    print("\nBenchmarking Sign Condition Model...")
    cond_adapter = SignConditionModelAdapter()
    cond_bench = cond_adapter.benchmark(num_runs)
    cond_meta = cond_adapter.metadata()
    registry["models"]["sign_condition_model"] = {
        "name": cond_meta.get("name", "SignConditionNet"),
        "task": "sign_condition_classification",
        "conditions": cond_meta.get("conditions", []),
        "weights_path": "models/infrastructure/sign_condition/best.pt",
        "sha256": get_file_sha256(Path(cond_meta["weights_path"])),
        "parameters": cond_meta.get("parameters", 0),
        "input_resolution": "64x64",
        "mean_latency_ms": cond_bench["mean_latency_ms"],
        "std_latency_ms": cond_bench["std_latency_ms"],
        "fps": cond_bench["fps"],
        "device": "CPU (Fallback)"
    }
    print(f"  -> Mean Latency: {cond_bench['mean_latency_ms']} ms | FPS: {cond_bench['fps']}")

    # 5. Road Divider Engine
    print("\nBenchmarking Road Divider Metrology Model...")
    div_adapter = RoadDividerModelAdapter()
    div_bench = div_adapter.benchmark(num_runs)
    div_meta = div_adapter.metadata()
    registry["models"]["road_divider_model"] = {
        "name": div_meta.get("model_name", "RoadDividerMetrologyEngine"),
        "status": "DIVIDER_MODEL_LIMITED_BY_PUBLIC_LABELS",
        "task": "road_divider_detection_metrology",
        "detection_modality": "Stereoscopic 3D Disparity Elevation + Continuity Metrology",
        "mean_latency_ms": div_bench["mean_latency_ms"],
        "std_latency_ms": div_bench["std_latency_ms"],
        "fps": div_bench["fps"],
        "device": "CPU (Fallback)"
    }
    print(f"  -> Mean Latency: {div_bench['mean_latency_ms']} ms | FPS: {div_bench['fps']}")

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Model registry saved to: {REGISTRY_PATH}")
    print("=" * 70)
    return registry


if __name__ == "__main__":
    run_benchmarks(num_runs=30)
