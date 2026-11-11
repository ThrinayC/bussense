#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Phase 3: Stereo Validation
Evaluates stereo depth against RSRD ground truth on smoke test (10 frames)
and comprehensive multi-sequence sample without hardcoded thresholds.
Outputs metrics to outputs/stereo_debug/stereo_metrics.json and 6-panel visualizations.
"""

import sys
import os
import json
import time
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from stereo_depth import get_stereo_matcher, disparity_to_depth
from inspect_rsrd import get_matching_calib, validate_calibration

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "RSRD"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stereo_debug"


def compute_stereo_metrics(pred_disp: np.ndarray, gt_disp: np.ndarray,
                           pred_depth: np.ndarray, gt_depth: np.ndarray) -> Dict[str, float]:
    """Computes quantitative metrics on valid GT pixels."""
    # Disparity evaluation mask: GT disp > 0 and pred disp > 0
    val_disp = (gt_disp > 0.5) & (pred_disp > 0) & np.isfinite(gt_disp) & np.isfinite(pred_disp)
    metrics = {}

    if np.sum(val_disp) > 10:
        d_gt = gt_disp[val_disp]
        d_pred = pred_disp[val_disp]
        abs_diff = np.abs(d_pred - d_gt)

        # End-Point Error (EPE)
        metrics["disparity_epe_px"] = float(np.mean(abs_diff))
        metrics["disparity_rmse_px"] = float(np.sqrt(np.mean(abs_diff ** 2)))

        # Bad pixel rate (D1: err > 3 px and relative err > 5%)
        bad_px = (abs_diff > 3.0) & ((abs_diff / d_gt) > 0.05)
        metrics["bad_pixel_rate_pct"] = float(np.mean(bad_px) * 100.0)
        metrics["valid_disparity_pixels"] = int(np.sum(val_disp))
    else:
        metrics["disparity_epe_px"] = float("nan")
        metrics["disparity_rmse_px"] = float("nan")
        metrics["bad_pixel_rate_pct"] = float("nan")
        metrics["valid_disparity_pixels"] = 0

    # Depth evaluation mask: GT depth > 0.5m and pred depth > 0.5m and < 50m
    val_depth = (gt_depth > 0.5) & (pred_depth > 0.5) & (gt_depth < 50.0) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
    if np.sum(val_depth) > 10:
        z_gt = gt_depth[val_depth]
        z_pred = pred_depth[val_depth]
        z_diff = np.abs(z_pred - z_gt)

        metrics["depth_mae_m"] = float(np.mean(z_diff))
        metrics["depth_rmse_m"] = float(np.sqrt(np.mean(z_diff ** 2)))
        metrics["depth_abs_rel"] = float(np.mean(z_diff / z_gt))
        metrics["valid_depth_pixels"] = int(np.sum(val_depth))
    else:
        metrics["depth_mae_m"] = float("nan")
        metrics["depth_rmse_m"] = float("nan")
        metrics["depth_abs_rel"] = float("nan")
        metrics["valid_depth_pixels"] = 0

    return metrics


def evaluate_dataset(
    data_dir: Path,
    output_dir: Path,
    matcher_name: str = "fast_foundation",
    num_frames: int = 30,
    smoke_test: bool = False
):
    output_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = data_dir / "calibration_files"

    is_half = (data_dir / "left_half").exists()
    left_dir = data_dir / ("left_half" if is_half else "left")
    right_dir = data_dir / ("right_half" if is_half else "right")
    disp_dir = data_dir / ("disparity_half" if is_half else "disparity")
    depth_dir = data_dir / ("depth_half" if is_half else "depth")

    left_files = sorted(list(left_dir.glob("*.jpg")) + list(left_dir.glob("*.png")))
    if not left_files:
        raise RuntimeError(f"No left images found in {left_dir}")

    total_available = len(left_files)
    if smoke_test:
        eval_count = min(10, total_available)
        stride = 1
    else:
        eval_count = min(num_frames, total_available)
        stride = max(1, total_available // eval_count)

    sample_indices = list(range(0, total_available, stride))[:eval_count]
    print("============================================================")
    print(f"STEREO EVALUATION — Mode: {'SMOKE TEST' if smoke_test else 'MULTI-SEQUENCE EVAL'}")
    print("============================================================")
    print(f"Model Backend      : {matcher_name}")
    print(f"Total Frames Avail : {total_available}")
    print(f"Evaluating Frames  : {len(sample_indices)}")

    # Initialize matcher
    t_start = time.time()
    matcher = get_stereo_matcher(matcher_name)
    load_time = time.time() - t_start
    print(f"Matcher initialization time: {load_time:.2f}s")

    all_metrics = []
    latencies = []

    # Process frames
    for idx, frame_idx in enumerate(sample_indices):
        left_path = left_files[frame_idx]
        stem = left_path.stem
        right_path = right_dir / left_path.name
        disp_path = disp_dir / f"{stem}.png"
        depth_path = depth_dir / f"{stem}.png"
        calib_path = get_matching_calib(stem, calib_dir, is_half=is_half)

        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        with open(calib_path, "rb") as f:
            calib = pickle.load(f)

        # Predict
        t0 = time.time()
        pred_disp = matcher.estimate_disparity(left_img, right_img)
        pred_depth, _ = disparity_to_depth(pred_disp, calib)
        latency = time.time() - t0
        latencies.append(latency)

        # Load GT
        gt_disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        gt_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0

        metrics = compute_stereo_metrics(pred_disp, gt_disp, pred_depth, gt_depth)
        metrics["frame_id"] = stem
        metrics["latency_s"] = float(latency)
        all_metrics.append(metrics)

        print(f"  Frame [{idx+1}/{len(sample_indices)}] {stem}: "
              f"EPE={metrics.get('disparity_epe_px', float('nan')):.2f} px | "
              f"BadPx={metrics.get('bad_pixel_rate_pct', float('nan')):.1f}% | "
              f"DepthMAE={metrics.get('depth_mae_m', float('nan')):.3f} m | "
              f"FPS={1.0/latency:.1f}")

        # Save 6-panel visualization for first 10 frames
        if idx < 10:
            fig, axes = plt.subplots(2, 3, figsize=(18, 9))

            # Row 1: Left RGB, GT Disparity, Predicted Disparity
            axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f"LEFT RGB ({stem})", fontsize=11)
            axes[0, 0].axis("off")

            vmax_d = np.percentile(gt_disp[gt_disp > 0], 98) if np.any(gt_disp > 0) else 64.0
            im_gtd = axes[0, 1].imshow(gt_disp, cmap="plasma", vmin=0, vmax=vmax_d)
            axes[0, 1].set_title("GT Disparity (px)", fontsize=11)
            axes[0, 1].axis("off")
            fig.colorbar(im_gtd, ax=axes[0, 1], fraction=0.035, pad=0.04)

            disp_clean = pred_disp.copy()
            disp_clean[disp_clean <= 0] = 0
            im_prd = axes[0, 2].imshow(disp_clean, cmap="plasma", vmin=0, vmax=vmax_d)
            axes[0, 2].set_title(f"PRED Disparity (EPE: {metrics.get('disparity_epe_px', 0):.2f} px)", fontsize=11)
            axes[0, 2].axis("off")
            fig.colorbar(im_prd, ax=axes[0, 2], fraction=0.035, pad=0.04)

            # Row 2: Right RGB, GT Depth, Predicted Depth
            axes[1, 0].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title("RIGHT RGB", fontsize=11)
            axes[1, 0].axis("off")

            vmax_z = min(30.0, np.percentile(gt_depth[gt_depth > 0], 95)) if np.any(gt_depth > 0) else 25.0
            im_gtz = axes[1, 1].imshow(gt_depth, cmap="turbo", vmin=0, vmax=vmax_z)
            axes[1, 1].set_title("GT Depth (m)", fontsize=11)
            axes[1, 1].axis("off")
            fig.colorbar(im_gtz, ax=axes[1, 1], fraction=0.035, pad=0.04)

            depth_clean = pred_depth.copy()
            im_prz = axes[1, 2].imshow(depth_clean, cmap="turbo", vmin=0, vmax=vmax_z)
            axes[1, 2].set_title(f"PRED Depth (MAE: {metrics.get('depth_mae_m', 0):.3f} m)", fontsize=11)
            axes[1, 2].axis("off")
            fig.colorbar(im_prz, ax=axes[1, 2], fraction=0.035, pad=0.04)

            plt.tight_layout()
            vis_path = output_dir / f"eval_frame_{idx:02d}_{stem}.png"
            plt.savefig(str(vis_path), dpi=120, bbox_inches="tight")
            plt.close()

    # Aggregate metrics
    valid_epes = [m["disparity_epe_px"] for m in all_metrics if not np.isnan(m["disparity_epe_px"])]
    valid_badpx = [m["bad_pixel_rate_pct"] for m in all_metrics if not np.isnan(m["bad_pixel_rate_pct"])]
    valid_maes = [m["depth_mae_m"] for m in all_metrics if not np.isnan(m["depth_mae_m"])]
    valid_rmses = [m["depth_rmse_m"] for m in all_metrics if not np.isnan(m["depth_rmse_m"])]
    valid_absrel = [m["depth_abs_rel"] for m in all_metrics if not np.isnan(m["depth_abs_rel"])]

    summary = {
        "model_name": matcher_name,
        "mode": "smoke_test" if smoke_test else "full_evaluation",
        "frames_evaluated": len(all_metrics),
        "mean_disparity_epe_px": float(np.mean(valid_epes)) if valid_epes else None,
        "median_disparity_epe_px": float(np.median(valid_epes)) if valid_epes else None,
        "mean_bad_pixel_rate_pct": float(np.mean(valid_badpx)) if valid_badpx else None,
        "mean_depth_mae_m": float(np.mean(valid_maes)) if valid_maes else None,
        "mean_depth_rmse_m": float(np.mean(valid_rmses)) if valid_rmses else None,
        "mean_depth_abs_rel": float(np.mean(valid_absrel)) if valid_absrel else None,
        "mean_latency_ms": float(np.mean(latencies) * 1000.0) if latencies else None,
        "mean_fps": float(1.0 / np.mean(latencies)) if latencies else None,
        "per_frame_metrics": all_metrics
    }

    metrics_file = output_dir / f"stereo_metrics_{matcher_name}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    # Also write default stereo_metrics.json
    with open(output_dir / "stereo_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Update comparison file
    comp_file = output_dir / "stereo_benchmark_comparison.json"
    comp_data = {}
    if comp_file.exists():
        try:
            with open(comp_file, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
        except Exception:
            pass
    comp_data[matcher_name] = {
        "model_name": summary["model_name"],
        "mode": summary["mode"],
        "frames_evaluated": summary["frames_evaluated"],
        "mean_disparity_epe_px": summary["mean_disparity_epe_px"],
        "median_disparity_epe_px": summary["median_disparity_epe_px"],
        "mean_bad_pixel_rate_pct": summary["mean_bad_pixel_rate_pct"],
        "mean_depth_mae_m": summary["mean_depth_mae_m"],
        "mean_depth_rmse_m": summary["mean_depth_rmse_m"],
        "mean_depth_abs_rel": summary["mean_depth_abs_rel"],
        "mean_latency_ms": summary["mean_latency_ms"],
        "mean_fps": summary["mean_fps"],
    }
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=4)

    print("\n------------------------------------------------------------")
    print("STEREO EVALUATION SUMMARY:")
    print("------------------------------------------------------------")
    print(f"Model                : {summary['model_name']}")
    print(f"Frames Evaluated     : {summary['frames_evaluated']}")
    print(f"Mean Disparity EPE   : {summary['mean_disparity_epe_px']:.3f} px")
    print(f"Median Disparity EPE : {summary['median_disparity_epe_px']:.3f} px")
    print(f"Mean Bad-Pixel Rate  : {summary['mean_bad_pixel_rate_pct']:.2f}%")
    print(f"Mean Depth MAE       : {summary['mean_depth_mae_m']:.4f} m ({summary['mean_depth_mae_m']*100:.2f} cm)")
    print(f"Mean Depth RMSE      : {summary['mean_depth_rmse_m']:.4f} m")
    print(f"Mean AbsRel Error    : {summary['mean_depth_abs_rel']:.4f}")
    print(f"Mean Inference Speed : {summary['mean_latency_ms']:.1f} ms ({summary['mean_fps']:.1f} FPS)")
    print(f"Results saved to     : {metrics_file}")
    print("============================================================")
    print("STEREO METRIC VALIDATION: COMPLETE")
    print("============================================================")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stereo depth on RSRD dataset.")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to RSRD dataset")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Path to save output artifacts")
    parser.add_argument("--model", type=str, default="fast_foundation", help="Stereo model to evaluate")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames for multi-sequence evaluation")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 10-frame smoke test")
    args = parser.parse_args()

    evaluate_dataset(
        Path(args.data_dir),
        Path(args.output_dir),
        matcher_name=args.model,
        num_frames=args.num_frames,
        smoke_test=args.smoke_test
    )
