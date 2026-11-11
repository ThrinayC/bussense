"""
BusSense — Real Hardware Benchmark Suite
========================================
Executes empirical benchmarks on the local execution machine.
Measures real latency percentiles (p50, p95, p99), FPS throughput,
CPU utilization, process RAM (MB), and device execution target.

Outputs:
  - Console summary table
  - benchmarks/hardware_benchmark_report.json
"""

import os
import sys
import time
import json
import platform
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from bussense_infrastructure.waterlogging_engine import WaterloggingEngine
from bussense_infrastructure.traffic_sign_engine import TrafficSignEngine
from bussense_infrastructure.zebra_crossing_engine import ZebraCrossingEngine
from bussense_infrastructure.road_divider_engine import RoadDividerEngine
from bussense_infrastructure.road_hazard_engine import RoadHazardEngine
from bussense_infrastructure.evidence_engine import EvidenceEngine
from bussense_infrastructure.multi_bus_fusion import MultiBusFusionEngine


def get_system_hardware_info() -> Dict[str, Any]:
    cuda_avail = torch.cuda.is_available()
    vram_mb = 0.0
    device_name = platform.processor() or "Unknown CPU"

    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    ram_total_mb = psutil.virtual_memory().total / (1024 * 1024)
    ram_avail_mb = psutil.virtual_memory().available / (1024 * 1024)

    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "processor": platform.processor(),
        "total_ram_mb": round(ram_total_mb, 1),
        "available_ram_mb": round(ram_avail_mb, 1),
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "vram_mb": round(vram_mb, 1),
        "execution_device": "CUDA (GPU)" if cuda_avail else "CPU (Fallback)",
        "benchmark_validity": "REAL_HARDWARE_MEASURED"
    }


def compute_latencies(durations: List[float]) -> Dict[str, float]:
    arr = np.array(durations) * 1000.0  # ms
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "fps": round(1000.0 / float(np.mean(arr)), 2) if np.mean(arr) > 0 else 0.0
    }


def benchmark_waterlogging(num_frames: int = 100) -> Dict[str, Any]:
    engine = WaterloggingEngine()
    durations = []
    process = psutil.Process()

    # Generate synthetic camera frames (640x480)
    H, W = 480, 640
    rgb = np.ones((H, W, 3), dtype=np.uint8) * 100
    road_mask = np.zeros((H, W), dtype=np.uint8)
    road_mask[240:, 100:540] = 255
    # Smooth wet patch
    rgb[350:420, 200:440] = [180, 180, 180]

    ram_start = process.memory_info().rss / (1024 * 1024)

    for i in range(num_frames):
        t0 = time.perf_counter()
        _ = engine.process_frame(
            rgb_img=rgb,
            frame_idx=i,
            timestamp=float(i * 0.1),
            road_mask=road_mask
        )
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    ram_end = process.memory_info().rss / (1024 * 1024)
    stats = compute_latencies(durations)
    stats["ram_delta_mb"] = round(ram_end - ram_start, 2)
    stats["num_frames"] = num_frames
    stats["weights_status"] = "CORRIDOR_METROLOGY_HEURISTIC"
    return stats


def benchmark_traffic_signs(num_frames: int = 100) -> Dict[str, Any]:
    engine = TrafficSignEngine()
    durations = []
    process = psutil.Process()

    # 640x480 RGB frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    ram_start = process.memory_info().rss / (1024 * 1024)

    for i in range(num_frames):
        t0 = time.perf_counter()
        _ = engine.update(
            rgb_img=frame,
            timestamp=float(i * 0.1)
        )
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    ram_end = process.memory_info().rss / (1024 * 1024)
    stats = compute_latencies(durations)
    stats["ram_delta_mb"] = round(ram_end - ram_start, 2)
    stats["num_frames"] = num_frames
    stats["weights_status"] = "SYNTHETIC_TRACKER_TAXONOMY"
    return stats


def benchmark_zebra_crossing(num_frames: int = 100) -> Dict[str, Any]:
    engine = ZebraCrossingEngine()
    durations = []
    process = psutil.Process()

    # 640x480 RGB frame with zebra stripes
    H, W = 480, 640
    frame = np.ones((H, W, 3), dtype=np.uint8) * 80
    for stripe in range(250, 450, 40):
        frame[stripe:stripe+20, 150:490] = 230

    ram_start = process.memory_info().rss / (1024 * 1024)

    for i in range(num_frames):
        t0 = time.perf_counter()
        _ = engine.process_frame(
            rgb_img=frame,
            frame_idx=i,
            timestamp=float(i * 0.1)
        )
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    ram_end = process.memory_info().rss / (1024 * 1024)
    stats = compute_latencies(durations)
    stats["ram_delta_mb"] = round(ram_end - ram_start, 2)
    stats["num_frames"] = num_frames
    stats["weights_status"] = "PAINT_COVERAGE_METROLOGY"
    return stats


def benchmark_road_divider(num_frames: int = 100) -> Dict[str, Any]:
    engine = RoadDividerEngine()
    durations = []
    process = psutil.Process()

    H, W = 480, 640
    rgb = np.ones((H, W, 3), dtype=np.uint8) * 100

    ram_start = process.memory_info().rss / (1024 * 1024)

    for i in range(num_frames):
        t0 = time.perf_counter()
        _ = engine.evaluate_divider_track(
            observed_intervals=[(0.0, 30.0), (45.0, 1200.0)],
            expected_range_m=(0.0, 1241.7),
            allowed_openings=[{"start_m": 70.0, "end_m": 85.0, "type": "U_TURN"}],
            frame_idx=i,
            timestamp=float(i * 0.1)
        )
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    ram_end = process.memory_info().rss / (1024 * 1024)
    stats = compute_latencies(durations)
    stats["ram_delta_mb"] = round(ram_end - ram_start, 2)
    stats["num_frames"] = num_frames
    stats["weights_status"] = "PLANE_METROLOGY_HEURISTIC"
    return stats


def benchmark_evidence_and_fusion(num_observations: int = 500) -> Dict[str, Any]:
    evidence_eng = EvidenceEngine()
    fusion_eng = MultiBusFusionEngine()
    process = psutil.Process()

    ram_start = process.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()

    for i in range(num_observations):
        bus_id = f"BUS_DEMO_{1 + (i % 4):03d}"
        eval_res = evidence_eng.evaluate_missing_asset(
            asset_type="DIVIDER",
            expected_state="YES",
            observed_present=False,
            visibility_score=0.88,
            consecutive_absences=4,
            unique_buses=1 + (i % 3)
        )
        obs_dict = {
            "observation_id": f"OBS_BENCH_{i}",
            "event_type": "MISSING_DIVIDER_CANDIDATE",
            "road_segment_id": "SEG-TRICHY-BHARATHIDASAN-01",
            "latitude": 10.8030,
            "longitude": 78.6880,
            "heading": 325.0,
            "visibility_score": 0.88,
            "status": eval_res.status,
            "parameters": {"is_missing": True}
        }
        fusion_eng.ingest_bus_pass(
            bus_id=bus_id,
            pass_id=f"PASS_{bus_id}_{i//4}",
            observations=[obs_dict]
        )

    t1 = time.perf_counter()
    ram_end = process.memory_info().rss / (1024 * 1024)

    total_time = t1 - t0
    ops_per_sec = round(num_observations / total_time, 1)

    return {
        "num_observations": num_observations,
        "total_time_s": round(total_time, 3),
        "throughput_obs_per_sec": ops_per_sec,
        "mean_latency_ms": round((total_time / num_observations) * 1000.0, 3),
        "ram_delta_mb": round(ram_end - ram_start, 2),
        "active_clusters": len(fusion_eng.clusters)
    }


def run_full_hardware_benchmark() -> Dict[str, Any]:
    print("=" * 70)
    print("BUSSENSE — REAL HARDWARE BENCHMARK SUITE")
    print("=" * 70)

    hw_info = get_system_hardware_info()
    print(f"OS: {hw_info['os']}")
    print(f"CPU: {hw_info['processor']} ({hw_info['cpu_count_logical']} logical cores)")
    print(f"RAM: {hw_info['total_ram_mb']} MB (Available: {hw_info['available_ram_mb']} MB)")
    print(f"Execution Target: {hw_info['execution_device']}")
    print(f"CUDA Available: {hw_info['cuda_available']}")
    print(f"Validity: {hw_info['benchmark_validity']}")
    print("-" * 70)

    print("\n[1/5] Benchmarking Waterlogging Extent Engine (100 frames)...")
    water_stats = benchmark_waterlogging(100)
    print(f"      -> Mean: {water_stats['mean_ms']} ms | p50: {water_stats['p50_ms']} ms | p95: {water_stats['p95_ms']} ms | FPS: {water_stats['fps']}")

    print("\n[2/5] Benchmarking Traffic Sign Tracking Engine (100 frames)...")
    sign_stats = benchmark_traffic_signs(100)
    print(f"      -> Mean: {sign_stats['mean_ms']} ms | p50: {sign_stats['p50_ms']} ms | p95: {sign_stats['p95_ms']} ms | FPS: {sign_stats['fps']}")

    print("\n[3/5] Benchmarking Zebra Crossing Paint Metrology (100 frames)...")
    zebra_stats = benchmark_zebra_crossing(100)
    print(f"      -> Mean: {zebra_stats['mean_ms']} ms | p50: {zebra_stats['p50_ms']} ms | p95: {zebra_stats['p95_ms']} ms | FPS: {zebra_stats['fps']}")

    print("\n[4/5] Benchmarking Road Divider Metrology (100 frames)...")
    divider_stats = benchmark_road_divider(100)
    print(f"      -> Mean: {divider_stats['mean_ms']} ms | p50: {divider_stats['p50_ms']} ms | p95: {divider_stats['p95_ms']} ms | FPS: {divider_stats['fps']}")

    print("\n[5/5] Benchmarking Evidence & Multi-Bus Fusion Engine (500 obs)...")
    fusion_stats = benchmark_evidence_and_fusion(500)
    print(f"      -> Throughput: {fusion_stats['throughput_obs_per_sec']} obs/sec | Mean latency: {fusion_stats['mean_latency_ms']} ms")

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw_info,
        "benchmarks": {
            "waterlogging_engine": water_stats,
            "traffic_sign_engine": sign_stats,
            "zebra_crossing_engine": zebra_stats,
            "road_divider_engine": divider_stats,
            "evidence_and_fusion_engine": fusion_stats
        }
    }

    # Save to json
    out_dir = ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "hardware_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE. Saved empirical report to: {report_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_full_hardware_benchmark()
