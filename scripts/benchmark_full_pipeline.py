#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/benchmark_full_pipeline.py
==================================
Empirical Host Hardware Benchmarks for BusSense Perception Pipeline (Section Q).

Measures and records:
- Host hardware specs (CPU, cores, RAM, OS, Python, PyTorch)
- Individual model / branch benchmarks (mean, p50, p95, FPS)
- End-to-end pipeline benchmark (mean, p50, p95, p99, FPS, RAM usage)
- Saves outputs/metrics/model_benchmarks.json and outputs/metrics/pipeline_benchmark.json
"""

import os
import sys
import time
import json
import platform
import psutil
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bussense_infrastructure.pipeline import BusSensePipeline, StereoInferenceManager


def get_hardware_info() -> Dict[str, Any]:
    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
    }


def benchmark_stats(latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    return {
        "samples": len(arr),
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(float(np.std(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "throughput_fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


def run_benchmarks(num_warmup: int = 5, num_runs: int = 25):
    print("=" * 70)
    print("BUSSENSE EMPIRICAL HOST HARDWARE BENCHMARKS")
    print("=" * 70)

    hw_info = get_hardware_info()
    print(f"Host OS: {hw_info['os']}")
    print(f"CPU: {hw_info['processor']} ({hw_info['cpu_count_logical']} cores)")
    print(f"RAM: {hw_info['total_ram_gb']} GB")
    print(f"Device: {hw_info['device']}")

    # Create dummy images representing 1080p camera inputs
    left_img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

    # Initialize Pipeline
    pipeline = BusSensePipeline()

    # 1. Benchmark Stereo Matcher
    print("\n[1/8] Benchmarking Stereo Matcher (Single-Pass SGBM)...")
    stereo_latencies = []
    for i in range(num_warmup + num_runs):
        t0 = time.perf_counter()
        pipeline.stereo_mgr.compute_stereo(left_img, right_img)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= num_warmup:
            stereo_latencies.append(dt)
    stereo_stats = benchmark_stats(stereo_latencies)
    print(f"  -> SGBM: {stereo_stats['mean_ms']} ms ({stereo_stats['throughput_fps']} FPS)")

    # 2. Benchmark YOLOv8n (Vehicle & Pedestrian)
    print("\n[2/8] Benchmarking YOLOv8n Detector...")
    yolo_latencies = []
    if pipeline.yolo_model is not None:
        for i in range(num_warmup + num_runs):
            t0 = time.perf_counter()
            pipeline.yolo_model.predict(left_img, conf=0.25, verbose=False)
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= num_warmup:
                yolo_latencies.append(dt)
        yolo_stats = benchmark_stats(yolo_latencies)
    else:
        yolo_stats = {"status": "NOT_AVAILABLE"}
    print(f"  -> YOLOv8n: {yolo_stats.get('mean_ms', 'N/A')} ms")

    # 3. Benchmark Defect Segmenter (Pothole & Crack)
    print("\n[3/8] Benchmarking Defect Segmentation (best.pt)...")
    seg_latencies = []
    if pipeline.seg_model is not None:
        for i in range(num_warmup + num_runs):
            t0 = time.perf_counter()
            pipeline.seg_model.predict(left_img, conf=0.25, verbose=False)
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= num_warmup:
                seg_latencies.append(dt)
        seg_stats = benchmark_stats(seg_latencies)
    else:
        seg_stats = {"status": "NOT_AVAILABLE"}
    print(f"  -> Defect Seg: {seg_stats.get('mean_ms', 'N/A')} ms")

    # 4. Benchmark Water Model Adapter
    print("\n[4/8] Benchmarking Water Model Adapter...")
    water_latencies = []
    for i in range(num_warmup + num_runs):
        t0 = time.perf_counter()
        pipeline.water_adapter.predict(left_img)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= num_warmup:
            water_latencies.append(dt)
    water_stats = benchmark_stats(water_latencies)
    print(f"  -> Water Seg: {water_stats['mean_ms']} ms")

    # 5. Benchmark Zebra Model Adapter
    print("\n[5/8] Benchmarking Zebra Crossing Model Adapter...")
    zebra_latencies = []
    for i in range(num_warmup + num_runs):
        t0 = time.perf_counter()
        pipeline.zebra_adapter.predict(left_img)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= num_warmup:
            zebra_latencies.append(dt)
    zebra_stats = benchmark_stats(zebra_latencies)
    print(f"  -> Zebra Seg: {zebra_stats['mean_ms']} ms")

    # 6. Benchmark Sign Model Adapter
    print("\n[6/8] Benchmarking Traffic Sign Model Adapter...")
    sign_latencies = []
    for i in range(num_warmup + num_runs):
        t0 = time.perf_counter()
        pipeline.sign_model_adapter.predict(left_img)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= num_warmup:
            sign_latencies.append(dt)
    sign_stats = benchmark_stats(sign_latencies)
    print(f"  -> Sign Det: {sign_stats['mean_ms']} ms")

    # 7. Benchmark Sign Condition Adapter
    print("\n[7/8] Benchmarking Sign Condition Adapter...")
    cond_latencies = []
    crop = left_img[100:228, 100:228]
    for i in range(num_warmup + num_runs):
        t0 = time.perf_counter()
        pipeline.sign_condition_adapter.predict(crop)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= num_warmup:
            cond_latencies.append(dt)
    cond_stats = benchmark_stats(cond_latencies)
    print(f"  -> Sign Condition: {cond_stats['mean_ms']} ms")

    # 8. Benchmark Divider Model Adapter
    print("\n[8/8] Benchmarking Road Divider Adapter...")
    div_latencies = []
    for i in range(num_warmup + num_runs):
        t0 = time.perf_counter()
        pipeline.divider_adapter.predict(left_img)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= num_warmup:
            div_latencies.append(dt)
    div_stats = benchmark_stats(div_latencies)
    print(f"  -> Divider: {div_stats['mean_ms']} ms")

    # Model benchmarks summary
    model_benchmarks = {
        "hardware_environment": hw_info,
        "models": {
            "stereo_sgbm_opencv": stereo_stats,
            "vehicle_pedestrian_yolov8n": yolo_stats,
            "pothole_crack_yolo_seg": seg_stats,
            "water_segnet_fred": water_stats,
            "zebra_segnet_ceymo": zebra_stats,
            "traffic_sign_detector_mtsd": sign_stats,
            "sign_condition_classifier": cond_stats,
            "road_divider_metrology": div_stats,
        }
    }

    metrics_dir = ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "model_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(model_benchmarks, f, indent=2)

    # 9. Full End-to-End Pipeline Benchmark
    print("\n" + "=" * 50)
    print("BENCHMARKING FULL END-TO-END PIPELINE")
    print("=" * 50)

    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    e2e_latencies = []
    for i in range(num_warmup + num_runs):
        t0 = time.perf_counter()
        pipeline.process_stereo_frame(left_img, right_img, frame_idx=i, timestamp=i*0.1)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= num_warmup:
            e2e_latencies.append(dt)

    ram_after_mb = process.memory_info().rss / (1024 * 1024)
    e2e_stats = benchmark_stats(e2e_latencies)
    e2e_stats["ram_usage_mb"] = round(ram_after_mb, 2)
    e2e_stats["ram_delta_mb"] = round(ram_after_mb - ram_before_mb, 2)

    pipeline_benchmark = {
        "hardware_environment": hw_info,
        "input_resolution": "1920x1080",
        "pipeline_benchmark": e2e_stats,
        "branch_latencies_summary": {
            "stereo_matcher_pct": round((stereo_stats["mean_ms"] / e2e_stats["mean_ms"]) * 100, 1),
            "yolo_detector_pct": round((yolo_stats.get("mean_ms", 0) / e2e_stats["mean_ms"]) * 100, 1) if yolo_stats.get("mean_ms") else 0,
            "segmenter_pct": round((seg_stats.get("mean_ms", 0) / e2e_stats["mean_ms"]) * 100, 1) if seg_stats.get("mean_ms") else 0,
            "infrastructure_adapters_pct": round(
                ((water_stats["mean_ms"] + zebra_stats["mean_ms"] + sign_stats["mean_ms"] + div_stats["mean_ms"]) / e2e_stats["mean_ms"]) * 100, 1
            )
        }
    }

    with open(metrics_dir / "pipeline_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_benchmark, f, indent=2)

    print(f"\n[BENCHMARK RESULT] End-to-End Pipeline Mean Latency: {e2e_stats['mean_ms']} ms ({e2e_stats['throughput_fps']} FPS)")
    print(f"Saved benchmarks to {metrics_dir / 'model_benchmarks.json'} and {metrics_dir / 'pipeline_benchmark.json'}")


if __name__ == "__main__":
    run_benchmarks(num_warmup=2, num_runs=10)
