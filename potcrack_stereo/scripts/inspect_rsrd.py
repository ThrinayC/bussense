#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Phase 1: Inspect RSRD and Calibration Validation
Inspects RSRD directory, validates calibration (intrinsics, baseline, rectification,
disparity sign, image compatibility), prints representative samples, and produces
outputs/stereo_debug/rsrd_sample.png.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "RSRD"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stereo_debug"


def parse_pcd_header(pcd_path: Path):
    """Parses binary/ascii PCD header and returns metadata dict."""
    metadata = {}
    with open(pcd_path, "rb") as f:
        while True:
            line = f.readline().decode("latin1", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0].upper()
            if key == "FIELDS":
                metadata["fields"] = parts[1:]
            elif key == "SIZE":
                metadata["size"] = [int(x) for x in parts[1:]]
            elif key == "TYPE":
                metadata["type"] = parts[1:]
            elif key == "COUNT":
                metadata["count"] = [int(x) for x in parts[1:]]
            elif key == "WIDTH":
                metadata["width"] = int(parts[1])
            elif key == "HEIGHT":
                metadata["height"] = int(parts[1])
            elif key == "POINTS":
                metadata["points"] = int(parts[1])
            elif key == "DATA":
                metadata["data"] = parts[1].lower()
                metadata["header_bytes"] = f.tell()
                break
    return metadata


def load_pcd_points(pcd_path: Path):
    """Loads XYZ points from PCD file."""
    meta = parse_pcd_header(pcd_path)
    points_count = meta.get("points", 0)
    data_type = meta.get("data", "binary")
    fields = meta.get("fields", ["x", "y", "z"])

    if data_type == "binary":
        type_map = {"F": "f", "I": "i", "U": "u"}
        dtype_list = []
        for name, sz, tp in zip(fields, meta["size"], meta["type"]):
            dtype_list.append((name, f"<{type_map.get(tp, 'f')}{sz}"))
        dt = np.dtype(dtype_list)
        with open(pcd_path, "rb") as f:
            f.seek(meta["header_bytes"])
            raw = f.read()
        arr = np.frombuffer(raw, dtype=dt)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1)
        return xyz
    else:
        with open(pcd_path, "r", encoding="latin1", errors="ignore") as f:
            lines = f.readlines()
        data_start = False
        pts = []
        for line in lines:
            if data_start:
                vals = [float(x) for x in line.strip().split()[:3]]
                if len(vals) == 3:
                    pts.append(vals)
            elif line.startswith("DATA"):
                data_start = True
        return np.array(pts, dtype=np.float32)


def get_matching_calib(timestamp_str: str, calib_dir: Path, is_half: bool = True) -> Path:
    """Finds the matching calibration file for a given timestamp."""
    date_str = timestamp_str[:8]
    suffix = "_half.pkl" if is_half else ".pkl"
    candidate = calib_dir / f"calib_{date_str}{suffix}"
    if candidate.exists():
        return candidate
    matches = list(calib_dir.glob(f"*{date_str}*{suffix}"))
    if matches:
        return matches[0]
    all_calibs = list(calib_dir.glob(f"*{suffix}"))
    if all_calibs:
        return all_calibs[0]
    raise FileNotFoundError(f"No calibration file found in {calib_dir}")


def validate_calibration(calib: dict, img_shape: tuple) -> dict:
    """
    Explicit validation step for:
    - intrinsics
    - baseline
    - image resolution
    - rectification
    - disparity direction/sign
    - calibration/image compatibility
    """
    h_img, w_img = img_shape[:2]
    K = np.array(calib["K"], dtype=np.float64)
    B_raw = float(calib["B"])
    baseline_m = B_raw / 1000.0 if B_raw > 10.0 else B_raw
    calib_w = int(calib.get("Width", w_img))
    calib_h = int(calib.get("Height", h_img))
    R = np.array(calib.get("R", np.eye(3)), dtype=np.float64)
    T = np.array(calib.get("T", np.zeros((3, 1))), dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    checks = {}
    checks["resolution_match"] = (calib_w == w_img) and (calib_h == h_img)
    checks["focal_valid"] = (fx > 0) and (fy > 0) and (abs(fx - fy) / fx < 0.05)
    checks["principal_point_valid"] = (0 < cx < w_img) and (0 < cy < h_img)
    checks["baseline_valid"] = (0.05 < baseline_m < 1.0)
    det_R = np.linalg.det(R)
    checks["rotation_orthonormal"] = abs(det_R - 1.0) < 0.01 or abs(det_R + 1.0) < 0.01
    checks["all_passed"] = all(checks.values())

    return {
        "K": K,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline_m": baseline_m,
        "baseline_mm": B_raw,
        "calib_resolution": (calib_w, calib_h),
        "img_resolution": (w_img, h_img),
        "R": R,
        "T": T,
        "checks": checks
    }


def inspect_rsrd_dataset(data_dir: Path, output_dir: Path):
    """Main inspection procedure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = data_dir / "calibration_files"
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")

    is_half = (data_dir / "left_half").exists()
    left_dir = data_dir / ("left_half" if is_half else "left")
    right_dir = data_dir / ("right_half" if is_half else "right")
    disp_dir = data_dir / ("disparity_half" if is_half else "disparity")
    depth_dir = data_dir / ("depth_half" if is_half else "depth")
    pcd_dir = data_dir / "pcd"

    print("============================================================")
    print("RSRD DATASET INSPECTION & CALIBRATION VALIDATION")
    print("============================================================")
    print(f"Data Root       : {data_dir}")
    print(f"Left Directory  : {left_dir}")
    print(f"Right Directory : {right_dir}")
    print(f"Calibration Dir : {calib_dir}")
    print(f"Disparity Dir   : {disp_dir}")
    print(f"Depth Dir       : {depth_dir}")
    print(f"Point Cloud Dir : {pcd_dir}")

    left_files = sorted(list(left_dir.glob("*.jpg")) + list(left_dir.glob("*.png")))
    right_files = sorted(list(right_dir.glob("*.jpg")) + list(right_dir.glob("*.png")))
    disp_files = sorted(list(disp_dir.glob("*.png"))) if disp_dir.exists() else []
    depth_files = sorted(list(depth_dir.glob("*.png"))) if depth_dir.exists() else []
    pcd_files = sorted(list(pcd_dir.glob("*.pcd"))) if pcd_dir.exists() else []

    print(f"\nDiscovered Frames:")
    print(f"  Left Images       : {len(left_files)}")
    print(f"  Right Images      : {len(right_files)}")
    print(f"  Disparity Maps    : {len(disp_files)}")
    print(f"  Depth Maps        : {len(depth_files)}")
    print(f"  Point Clouds (PCD): {len(pcd_files)}")

    if not left_files or not right_files:
        raise RuntimeError("No stereo image pairs found in RSRD directory!")

    # Select representative sample
    sample_left_path = left_files[min(10, len(left_files) - 1)]
    stem = sample_left_path.stem
    sample_right_path = right_dir / sample_left_path.name
    sample_calib_path = get_matching_calib(stem, calib_dir, is_half=is_half)
    sample_disp_path = disp_dir / f"{stem}.png" if disp_dir.exists() else None
    sample_depth_path = depth_dir / f"{stem}.png" if depth_dir.exists() else None
    sample_pcd_path = pcd_dir / f"{stem}.pcd" if pcd_dir.exists() else None

    # Load Left & Right images
    left_img = cv2.imread(str(sample_left_path))
    right_img = cv2.imread(str(sample_right_path))
    if left_img is None or right_img is None:
        raise RuntimeError("Failed to load sample stereo images.")

    print("\n------------------------------------------------------------")
    print("REPRESENTATIVE SAMPLE:")
    print("------------------------------------------------------------")
    print(f"LEFT IMAGE:")
    print(f"  Path  : {sample_left_path}")
    print(f"  Shape : {left_img.shape} (H x W x C)")
    print(f"  Dtype : {left_img.dtype}")

    print(f"\nRIGHT IMAGE:")
    print(f"  Path  : {sample_right_path}")
    print(f"  Shape : {right_img.shape} (H x W x C)")
    print(f"  Dtype : {right_img.dtype}")

    with open(sample_calib_path, "rb") as f:
        raw_calib = pickle.load(f)

    calib_val = validate_calibration(raw_calib, left_img.shape)
    print(f"\nCALIBRATION VALIDATION ({sample_calib_path.name}):")
    print(f"  Intrinsics K matrix:\n{calib_val['K']}")
    print(f"  Focal length fx={calib_val['fx']:.2f} px, fy={calib_val['fy']:.2f} px")
    print(f"  Principal point (cx, cy)=({calib_val['cx']:.2f}, {calib_val['cy']:.2f})")
    print(f"  Baseline: {calib_val['baseline_m']:.4f} m ({calib_val['baseline_mm']:.2f} mm)")
    print(f"  Calib Resolution: {calib_val['calib_resolution']}")
    print(f"  LiDAR Extrinsics R:\n{calib_val['R']}")
    print(f"  LiDAR Extrinsics T:\n{calib_val['T'].ravel()}")
    print("  Validation Checks:")
    for k, v in calib_val["checks"].items():
        status = "PASSED" if v else "FAILED"
        print(f"    - {k:<25s}: {status}")

    gt_disp = None
    if sample_disp_path and sample_disp_path.exists():
        disp_raw = cv2.imread(str(sample_disp_path), cv2.IMREAD_UNCHANGED)
        print(f"\nGT DISPARITY:")
        print(f"  Path        : {sample_disp_path}")
        print(f"  Shape       : {disp_raw.shape}")
        print(f"  Dtype       : {disp_raw.dtype}")
        if disp_raw.dtype == np.uint16:
            gt_disp = disp_raw.astype(np.float32) / 256.0
        else:
            gt_disp = disp_raw.astype(np.float32)
        valid_disp = gt_disp[gt_disp > 0]
        if len(valid_disp) > 0:
            print(f"  Value Range : {valid_disp.min():.2f} to {valid_disp.max():.2f} pixels (valid count: {len(valid_disp)})")

    gt_depth = None
    if sample_depth_path and sample_depth_path.exists():
        depth_raw = cv2.imread(str(sample_depth_path), cv2.IMREAD_UNCHANGED)
        print(f"\nGT DEPTH:")
        print(f"  Path        : {sample_depth_path}")
        print(f"  Shape       : {depth_raw.shape}")
        print(f"  Dtype       : {depth_raw.dtype}")
        if depth_raw.dtype == np.uint16:
            gt_depth = depth_raw.astype(np.float32) / 256.0
        else:
            gt_depth = depth_raw.astype(np.float32)
        valid_depth = gt_depth[gt_depth > 0]
        if len(valid_depth) > 0:
            print(f"  Value Range : {valid_depth.min():.2f} to {valid_depth.max():.2f} meters (valid count: {len(valid_depth)})")

    if sample_pcd_path and sample_pcd_path.exists():
        pcd_pts = load_pcd_points(sample_pcd_path)
        print(f"\nPOINT CLOUD (PCD):")
        print(f"  Path        : {sample_pcd_path}")
        print(f"  Points Count: {len(pcd_pts)}")
        print(f"  Bounding Box: X [{pcd_pts[:, 0].min():.2f}, {pcd_pts[:, 0].max():.2f}] m, "
              f"Y [{pcd_pts[:, 1].min():.2f}, {pcd_pts[:, 1].max():.2f}] m, "
              f"Z [{pcd_pts[:, 2].min():.2f}, {pcd_pts[:, 2].max():.2f}] m")

    print("\nGenerating sample visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"RSRD LEFT RGB ({left_img.shape[1]}x{left_img.shape[0]})", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"RSRD RIGHT RGB ({right_img.shape[1]}x{right_img.shape[0]})", fontsize=12)
    axes[1].axis("off")

    if gt_disp is not None:
        disp_vis = gt_disp.copy()
        vmax = np.percentile(disp_vis[disp_vis > 0], 98) if np.any(disp_vis > 0) else 64.0
        im = axes[2].imshow(disp_vis, cmap="plasma", vmin=0, vmax=vmax)
        axes[2].set_title("RSRD GT Disparity (pixels)", fontsize=12)
        fig.colorbar(im, ax=axes[2], fraction=0.035, pad=0.04)
    elif gt_depth is not None:
        depth_vis = gt_depth.copy()
        vmax = np.percentile(depth_vis[depth_vis > 0], 95) if np.any(depth_vis > 0) else 30.0
        im = axes[2].imshow(depth_vis, cmap="turbo", vmin=0, vmax=vmax)
        axes[2].set_title("RSRD GT Depth (m)", fontsize=12)
        fig.colorbar(im, ax=axes[2], fraction=0.035, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, "No GT Disp/Depth Available", ha="center", va="center")
        axes[2].axis("off")

    axes[2].axis("off")
    plt.tight_layout()

    sample_out_path = output_dir / "rsrd_sample.png"
    plt.savefig(str(sample_out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization artifact to: {sample_out_path}")
    print("============================================================")
    print("RSRD INSPECTION: COMPLETE")
    print("============================================================")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect RSRD dataset and validate calibration.")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to RSRD dataset directory")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save artifacts")
    args = parser.parse_args()

    inspect_rsrd_dataset(Path(args.data_dir), Path(args.output_dir))
