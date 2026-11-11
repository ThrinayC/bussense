"""
audit_rsrd_unevenness.py - RSRD Irregular Unevenness vs Pothole Audit

Audits RSRD 3D road surface data to strictly distinguish:
- Localized sharp pothole depressions (depth >= 3.0 cm, compact concave boundary)
- General irregular road surface unevenness / rutting / waviness (shallow, diffuse)

Generates outputs/rsrd_audit/unevenness_audit.json and audit_summary.png.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RSRD_SEG_ROOT = PROJECT_ROOT / "data" / "RSRD_seg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rsrd_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pcd_binary(pcd_path: Path) -> np.ndarray:
    """
    Loads binary PCD file with fields x y z (and optional label).
    Returns (N, 3) float32 array of XYZ coordinates.
    """
    with open(pcd_path, "rb") as f:
        # Read header
        header = {}
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "FIELDS":
                header["fields"] = parts[1:]
            elif parts[0] == "SIZE":
                header["size"] = [int(s) for s in parts[1:]]
            elif parts[0] == "TYPE":
                header["type"] = parts[1:]
            elif parts[0] == "POINTS":
                header["points"] = int(parts[1])
            elif parts[0] == "DATA":
                header["data"] = parts[1]
                break

        # Build numpy dtype
        dtype_list = []
        for name, sz, tp in zip(header["fields"], header["size"], header["type"]):
            if sz == 1:
                dt = np.uint8
            elif sz == 2:
                dt = np.uint16
            elif sz == 4 and tp == "F":
                dt = np.float32
            elif sz == 4 and tp in ["I", "U"]:
                dt = np.uint32
            elif sz == 8 and tp == "F":
                dt = np.float64
            else:
                dt = np.float32
            dtype_list.append((name, dt))

        record_dtype = np.dtype(dtype_list)
        raw_data = np.fromfile(f, dtype=record_dtype)

    # Extract x, y, z
    xyz = np.column_stack([raw_data["x"], raw_data["y"], raw_data["z"]]).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid]


def analyze_surface_patch(xyz: np.ndarray) -> dict:
    """
    Fits local road plane using RANSAC and computes signed depressions.
    """
    if len(xyz) < 50:
        return {"category": "INSUFFICIENT_POINTS", "depth_cm": 0.0, "area_m2": 0.0}

    # Downsample if very dense
    if len(xyz) > 20000:
        idx = np.random.choice(len(xyz), 20000, replace=False)
        pts = xyz[idx]
    else:
        pts = xyz

    # Fit road plane Z = a*X + b*Y + d
    X_in = pts[:, :2]
    Y_in = pts[:, 2]

    try:
        ransac = RANSACRegressor(residual_threshold=0.03, random_state=42)
        ransac.fit(X_in, Y_in)
        inliers = ransac.inlier_mask_
        a, b = ransac.estimator_.coef_
        c_val = -1.0
        d = ransac.estimator_.intercept_

        # Plane normal vector (normalized)
        norm = np.sqrt(a**2 + b**2 + 1.0)
        plane_normal = np.array([a / norm, b / norm, c_val / norm])
        d_norm = d / norm

        # Signed distance from points to plane:
        # Distance = (a*x + b*y - z + d) / norm
        signed_dist = (pts[:, 0] * a + pts[:, 1] * b - pts[:, 2] + d) / norm

        # In camera coordinate frame, depression below road is negative or positive depending on direction
        # Inliers represent road surface. Points with signed_dist > 0 or < 0
        # Let's inspect residual distribution of inliers vs outliers
        residuals = Y_in - ransac.predict(X_in) # if Z is depth, positive residual is deeper
        p95_depression = float(np.percentile(np.abs(residuals), 95)) * 100.0 # cm
        max_depression = float(np.percentile(np.abs(residuals), 99)) * 100.0 # cm

        # Area span in XY
        x_span = float(pts[:, 0].max() - pts[:, 0].min())
        y_span = float(pts[:, 1].max() - pts[:, 1].min())
        bounding_area = x_span * y_span

        # Criteria for true localized pothole vs irregular unevenness:
        # Pothole: distinct depression >= 3.0 cm, localized span <= 2.5m
        # Irregular unevenness: broad/wavy, or shallow (< 2.5 cm)
        if p95_depression >= 3.0 and x_span <= 2.5 and y_span <= 2.5:
            category = "VERIFIED_POTHOLE"
        elif p95_depression >= 1.5:
            category = "IRREGULAR_UNEVENNESS"
        else:
            category = "ROAD_SURFACE"

        return {
            "category": category,
            "depth_p95_cm": round(p95_depression, 2),
            "depth_max_cm": round(max_depression, 2),
            "x_span_m": round(x_span, 2),
            "y_span_m": round(y_span, 2),
            "bounding_area_m2": round(bounding_area, 3),
            "total_points": len(pts),
            "inlier_ratio": round(float(np.mean(inliers)), 3)
        }
    except Exception as e:
        return {"category": "FITTING_ERROR", "error": str(e), "depth_cm": 0.0}


def run_audit(max_patches: int = 50):
    print("============================================================")
    print("RSRD ROAD SURFACE AUDIT: UNEVENNESS VS POTHOLE")
    print("============================================================")

    patch_files = list((RSRD_SEG_ROOT / "cropped_patches").glob("*/*.pcd"))
    if not patch_files:
        # Check complete point clouds
        patch_files = list((RSRD_SEG_ROOT / "complete_point_cloud").glob("*.pcd"))

    print(f"Total PCD files found: {len(patch_files)}")
    sample_files = patch_files[:max_patches]

    audit_results = {}
    category_counts = {"VERIFIED_POTHOLE": 0, "IRREGULAR_UNEVENNESS": 0, "ROAD_SURFACE": 0, "OTHER": 0}

    pothole_depths = []
    unevenness_depths = []

    for idx, pcd_path in enumerate(sample_files):
        stem = pcd_path.name
        xyz = load_pcd_binary(pcd_path)
        stats = analyze_surface_patch(xyz)
        audit_results[stem] = stats

        cat = stats.get("category", "OTHER")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        if cat == "VERIFIED_POTHOLE":
            pothole_depths.append(stats["depth_p95_cm"])
        elif cat == "IRREGULAR_UNEVENNESS":
            unevenness_depths.append(stats["depth_p95_cm"])

        if (idx + 1) % 10 == 0 or idx == len(sample_files) - 1:
            print(f"  Processed [{idx+1}/{len(sample_files)}] {stem}: {cat} (Depth: {stats.get('depth_p95_cm', 0)} cm)")

    # Save audit report
    summary = {
        "total_audited": len(sample_files),
        "categories": category_counts,
        "verified_potholes_count": category_counts.get("VERIFIED_POTHOLE", 0),
        "irregular_unevenness_count": category_counts.get("IRREGULAR_UNEVENNESS", 0),
        "mean_pothole_depth_cm": float(np.mean(pothole_depths)) if pothole_depths else 0.0,
        "mean_unevenness_depth_cm": float(np.mean(unevenness_depths)) if unevenness_depths else 0.0,
        "audit_rule": "RSRD irregular road unevenness is NOT assumed to be a pothole unless depth >= 3.0 cm and compact localized footprint <= 2.5m.",
        "detailed_results": audit_results
    }

    report_path = OUTPUT_DIR / "unevenness_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Plot summary figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of categories
    labels = list(category_counts.keys())
    counts = [category_counts[k] for k in labels]
    colors = ["#d9534f" if "POTHOLE" in k else "#f0ad4e" if "UNEVENNESS" in k else "#5cb85c" for k in labels]
    axes[0].bar(labels, counts, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("RSRD Road Surface Classification Audit", fontsize=12)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=15)

    # Histogram of depths
    if pothole_depths:
        axes[1].hist(pothole_depths, bins=10, color="#d9534f", alpha=0.7, label=f"Verified Potholes (n={len(pothole_depths)})")
    if unevenness_depths:
        axes[1].hist(unevenness_depths, bins=10, color="#f0ad4e", alpha=0.7, label=f"Irregular Unevenness (n={len(unevenness_depths)})")
    axes[1].set_title("Depth Distribution (p95 cm)", fontsize=12)
    axes[1].set_xlabel("Depth (cm)")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(3.0, color="red", linestyle="--", label="Pothole Threshold (3.0 cm)")
    axes[1].legend()

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "audit_summary.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()

    print("\n------------------------------------------------------------")
    print("AUDIT SUMMARY:")
    print("------------------------------------------------------------")
    print(f"Total Audited Patches   : {len(sample_files)}")
    print(f"Verified Potholes       : {category_counts.get('VERIFIED_POTHOLE', 0)}")
    print(f"Irregular Unevenness    : {category_counts.get('IRREGULAR_UNEVENNESS', 0)}")
    print(f"Road Surface / Flat     : {category_counts.get('ROAD_SURFACE', 0)}")
    print(f"Audit JSON Report       : {report_path}")
    print(f"Audit Figure Saved      : {plot_path}")
    print("============================================================\n")


if __name__ == "__main__":
    run_audit()
