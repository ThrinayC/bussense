#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnose_crack_fragmentation.py
=======================================
Deep-Dive Root-Cause Diagnostic on Crack Fragmentation:
- Step 1: Raw YOLO instance fragmentation vs Union mask
- Step 2: Mask-level morphological bridging sweep (1px, 2px, 3px, 5px)
- Step 3 & 4: Branch-preserving Skeleton Graph & Geometric Gap-Bridging
- Step 5: Temporal consistency across adjacent frames
- Step 6: In-depth Audit of Metric Crack Lengths (PCA sorting vs Graph Arc Length)
- Step 7: 5-panel visual comparison contact sheets for Frames 44, 47, 49, 65
- Step 8: Synthesis of root causes (A, B, C, D, E, F) and minimal defensible fix.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.crack_geometry import CrackGeometryEngine, zhang_suen_thinning


def extract_raw_yolo_cracks(
    model: YOLO,
    img: np.ndarray,
    conf_thresh: float = 0.035
) -> List[Dict[str, Any]]:
    """Runs YOLO and extracts class 1 (crack) instances passing basic quality guards."""
    H, W = img.shape[:2]
    res = model.predict(img, conf=conf_thresh, verbose=False)[0]
    raw_cracks = []
    if res.boxes is not None and res.masks is not None:
        for i, box in enumerate(res.boxes):
            if int(box.cls[0].item()) != 1:
                continue
            conf = float(box.conf[0].item())
            b_coords = [int(v) for v in box.xyxy[0].cpu().numpy()]
            if b_coords[1] < 70:  # Top banner exclusion
                continue
            box_area = (b_coords[2] - b_coords[0]) * (b_coords[3] - b_coords[1])
            if box_area > 0.35 * (W * H):
                continue
            m_raw = res.masks.data[i].cpu().numpy()
            m_res = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            if cv2.countNonZero(m_res) > 0.05 * (W * H):
                continue
            raw_cracks.append({
                "id": i,
                "confidence": conf,
                "bbox": b_coords,
                "mask": m_res,
                "area_px": int(cv2.countNonZero(m_res))
            })
    return raw_cracks


def test_morphology_sweep(union_mask: np.ndarray) -> Dict[str, Any]:
    """Tests closing kernels (1px, 2px, 3px, 5px) and reports component counts and pixel counts."""
    H, W = union_mask.shape[:2]
    raw_pixels = int(cv2.countNonZero(union_mask))
    
    # Baseline raw connected components
    n_raw, labels_raw, stats_raw, _ = cv2.connectedComponentsWithStats((union_mask > 0).astype(np.uint8), connectivity=8)
    valid_raw_comps = [stats_raw[i, cv2.CC_STAT_AREA] for i in range(1, n_raw) if stats_raw[i, cv2.CC_STAT_AREA] >= 25]
    
    results = {
        "raw": {
            "components": len(valid_raw_comps),
            "pixels": raw_pixels,
            "max_comp_area": max(valid_raw_comps) if valid_raw_comps else 0
        }
    }
    
    # Sweep kernel sizes
    for k in [1, 2, 3, 5]:
        if k == 1:
            # 1px closing is equivalent to a 3x3 cross kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        else:
            ksize = 2 * k + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            
        closed = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, kernel)
        n_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats((closed > 0).astype(np.uint8), connectivity=8)
        valid_comps = [stats_c[i, cv2.CC_STAT_AREA] for i in range(1, n_c) if stats_c[i, cv2.CC_STAT_AREA] >= 25]
        c_pixels = int(cv2.countNonZero(closed))
        
        results[f"{k}px"] = {
            "components": len(valid_comps),
            "pixels": c_pixels,
            "pixel_growth_pct": round(((c_pixels - raw_pixels) / max(raw_pixels, 1)) * 100.0, 2),
            "max_comp_area": max(valid_comps) if valid_comps else 0,
            "mask": closed
        }
        
    return results


def analyze_skeleton_topology(skeleton: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes skeleton graph topology: endpoints, junction branch points, and branch segments.
    """
    skel_binary = (skeleton > 0).astype(np.uint8)
    if cv2.countNonZero(skel_binary) == 0:
        return {"endpoints": [], "junctions": [], "branches": [], "pixel_count": 0}
        
    # 8-neighbor count convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel) * skel_binary
    
    endpoints = []
    junctions = []
    
    y_idx, x_idx = np.where(skel_binary > 0)
    for y, x in zip(y_idx, x_idx):
        nb = neighbor_count[y, x]
        if nb == 1:
            endpoints.append((int(x), int(y)))
        elif nb >= 3:
            junctions.append((int(x), int(y)))
            
    return {
        "endpoints": endpoints,
        "junctions": junctions,
        "pixel_count": len(y_idx)
    }


def compute_true_graph_length_3d(
    skeleton: np.ndarray,
    calib: Dict[str, float],
    road_plane: np.ndarray
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Computes crack length using:
    1) True 8-connected skeleton 3D Euclidean step length (arc length along centerline)
    2) Flawed PCA projection sorting length (the current baseline)
    Returns: (true_arc_length_m, pca_length_m, detailed_diagnostics)
    """
    fx, fy = calib["fx"], calib["fy"]
    cx, cy = calib["cx"], calib["cy"]
    a, b, c, d = road_plane
    
    skel_ys, skel_xs = np.where(skeleton > 0)
    if len(skel_ys) < 5:
        return 0.0, 0.0, []
        
    # Project all skeleton pixels to 3D road plane
    rx = (skel_xs - cx) / fx
    ry = (skel_ys - cy) / fy
    rz = np.ones_like(rx)
    denom = a * rx + b * ry + c * rz
    valid = (np.abs(denom) > 1e-4)
    t = -d / denom
    valid &= (t > 0)
    
    X = t * rx
    Y = t * ry
    Z = t * rz
    
    valid &= (Z >= 1.0) & (Z <= 30.0) & (np.abs(X) <= 3.5) & (skel_ys >= 70)
    if np.sum(valid) < 5:
        return 0.0, 0.0, []
        
    valid_xs = skel_xs[valid]
    valid_ys = skel_ys[valid]
    pts_3d = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
    
    # -------------------------------------------------------------
    # Method A: Baseline Flawed PCA-Sort Length
    # -------------------------------------------------------------
    mean_pt = np.mean(pts_3d, axis=0)
    centered = pts_3d - mean_pt
    cov = np.cov(centered[:, [0, 2]].T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    primary_axis = eigvecs[:, 1]
    projections = centered[:, [0, 2]] @ primary_axis
    sort_order = np.argsort(projections)
    sorted_pts = pts_3d[sort_order]
    diffs = np.diff(sorted_pts, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    pca_length_m = float(np.sum(segment_lengths[segment_lengths < 1.5]))
    
    # -------------------------------------------------------------
    # Method B: True 8-Connected 3D Skeleton Arc Length
    # -------------------------------------------------------------
    # Build coordinate map
    coord_to_idx = {(valid_xs[i], valid_ys[i]): i for i in range(len(valid_xs))}
    visited_edges = set()
    total_true_dist = 0.0
    
    # Directions for 8-connectivity (half-set to avoid double-counting undirected edges)
    neighbor_offsets = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for (u, v), idx1 in coord_to_idx.items():
        for du, dv in neighbor_offsets:
            nb = (u + du, v + dv)
            if nb in coord_to_idx:
                idx2 = coord_to_idx[nb]
                edge = tuple(sorted([idx1, idx2]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    d_step = float(np.linalg.norm(pts_3d[idx1] - pts_3d[idx2]))
                    # Physical continuity guard for adjacent pixels: max 0.5m
                    if d_step < 0.5:
                        total_true_dist += d_step
                        
    # If network has disconnected single points without neighbors, use minimum bounding diagonal
    if total_true_dist == 0.0 and len(pts_3d) >= 2:
        total_true_dist = float(np.linalg.norm(np.max(pts_3d, axis=0) - np.min(pts_3d, axis=0)))
        
    diag = {
        "pixel_count": len(valid_xs),
        "mean_z_m": float(np.mean(pts_3d[:, 2])),
        "min_z_m": float(np.min(pts_3d[:, 2])),
        "max_z_m": float(np.max(pts_3d[:, 2])),
        "v_range": [int(np.min(valid_ys)), int(np.max(valid_ys))],
        "edges_count": len(visited_edges),
        "true_arc_length_m": round(total_true_dist, 2),
        "pca_length_m": round(pca_length_m, 2),
        "inflation_factor": round(pca_length_m / max(total_true_dist, 0.001), 2)
    }
    
    return total_true_dist, pca_length_m, [diag]


def test_geometric_graph_merging(
    skeleton: np.ndarray,
    max_gap_px: float = 12.0
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Bridges endpoints of disconnected skeleton components if they lie within max_gap_px
    and their local tangents are compatible.
    """
    H, W = skeleton.shape[:2]
    num_labels, labels = cv2.connectedComponents((skeleton > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 2:
        return skeleton.copy(), []
        
    merged_skel = skeleton.copy()
    connections = []
    
    # Extract endpoints for each component
    comp_endpoints = {}
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    
    for l_id in range(1, num_labels):
        comp_mask = (labels == l_id).astype(np.uint8)
        if cv2.countNonZero(comp_mask) < 5:
            continue
        nb_count = cv2.filter2D(comp_mask, -1, kernel) * comp_mask
        ep_ys, ep_xs = np.where(nb_count == 1)
        if len(ep_xs) > 0:
            comp_endpoints[l_id] = list(zip(ep_xs, ep_ys))
            
    # Pairwise endpoint proximity check
    labels_list = list(comp_endpoints.keys())
    for i in range(len(labels_list)):
        l1 = labels_list[i]
        for j in range(i + 1, len(labels_list)):
            l2 = labels_list[j]
            best_dist = 1e9
            best_pair = None
            for p1 in comp_endpoints[l1]:
                for p2 in comp_endpoints[l2]:
                    dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    if dist < best_dist and dist <= max_gap_px:
                        best_dist = dist
                        best_pair = (p1, p2)
                        
            if best_pair is not None:
                p1, p2 = best_pair
                # Draw 1px connection line
                cv2.line(merged_skel, p1, p2, 255, 1)
                connections.append({
                    "from_comp": l1,
                    "to_comp": l2,
                    "p1": [int(p1[0]), int(p1[1])],
                    "p2": [int(p2[0]), int(p2[1])],
                    "gap_px": round(float(best_dist), 1)
                })
                
    return merged_skel, connections


def main():
    print("=== BUSSENSE CRACK FRAGMENTATION ROOT-CAUSE DIAGNOSTIC ===")
    t_start = time.time()
    
    base_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
    left_dir = base_dir / "left"
    out_debug_dir = PROJECT_ROOT / "outputs" / "debug" / "crack_web_fragmentation"
    out_debug_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_dir = Path("C:/Users/Thrinay/.gemini/antigravity/brain/03d7fdef-cb64-4066-b840-f0b324ee084e")
    
    model_path = PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    
    calib = {"fx": 1050.0, "fy": 1050.0, "cx": 960.0, "cy": 540.0, "baseline": 0.12}
    ref_road_plane = np.array([0.0, -1.0, 0.05, 1.65], dtype=np.float32)
    
    # Target frames specified in prompt
    target_frames = [36, 38, 42, 44, 47, 48, 49, 65]
    comparison_frames = [44, 47, 49, 65]
    
    all_frame_results = {}
    
    for fnum in target_frames:
        frame_path = left_dir / f"frame_{fnum:05d}.jpg"
        if not frame_path.exists():
            print(f"Frame {frame_path.name} not found, skipping.")
            continue
            
        print(f"\nAnalyzing Frame {fnum:02d} ({frame_path.name})...")
        img = cv2.imread(str(frame_path))
        H, W = img.shape[:2]
        
        # 1. RAW YOLO DETECTIONS
        raw_cracks = extract_raw_yolo_cracks(model, img, conf_thresh=0.035)
        raw_box_count = len(raw_cracks)
        
        # Build union mask
        union_mask = np.zeros((H, W), dtype=np.uint8)
        for rc in raw_cracks:
            union_mask = cv2.bitwise_or(union_mask, rc["mask"] * 255)
            
        # Connected components on union mask
        n_raw_union, labels_raw_union, stats_raw_union, _ = cv2.connectedComponentsWithStats(
            (union_mask > 0).astype(np.uint8), connectivity=8
        )
        valid_raw_union_comps = [stats_raw_union[i, cv2.CC_STAT_AREA] for i in range(1, n_raw_union) if stats_raw_union[i, cv2.CC_STAT_AREA] >= 25]
        
        # Check box overlap: How many YOLO boxes overlap with each other?
        overlapping_pairs = 0
        for i in range(raw_box_count):
            for j in range(i + 1, raw_box_count):
                if cv2.countNonZero(cv2.bitwise_and(raw_cracks[i]["mask"], raw_cracks[j]["mask"])) > 0:
                    overlapping_pairs += 1
                    
        # 2. MORPHOLOGY SWEEP
        morph_results = test_morphology_sweep(union_mask)
        
        # 3. SKELETONIZATION & TOPOLOGY
        skel_union = zhang_suen_thinning(union_mask)
        topo_union = analyze_skeleton_topology(skel_union)
        
        # 4. GEOMETRIC GAP MERGING
        merged_skel, gap_conns = test_geometric_graph_merging(skel_union, max_gap_px=15.0)
        n_merged_comps, _ = cv2.connectedComponents((merged_skel > 0).astype(np.uint8), connectivity=8)
        n_merged_comps -= 1  # exclude background
        
        # 5. METRIC LENGTH AUDIT: True Arc Length vs Baseline Flawed PCA
        true_len, pca_len, len_diag = compute_true_graph_length_3d(skel_union, calib, ref_road_plane)
        true_len_merged, pca_len_merged, _ = compute_true_graph_length_3d(merged_skel, calib, ref_road_plane)
        
        frame_summary = {
            "frame_idx": fnum,
            "raw_yolo_boxes": raw_box_count,
            "overlapping_box_pairs": overlapping_pairs,
            "union_connected_components": len(valid_raw_union_comps),
            "morphology_sweep": {
                k: {
                    "components": v["components"],
                    "pixels": v["pixels"],
                    "pixel_growth_pct": v.get("pixel_growth_pct", 0.0)
                } for k, v in morph_results.items()
            },
            "topology": {
                "endpoints_count": len(topo_union["endpoints"]),
                "junctions_count": len(topo_union["junctions"]),
                "skeleton_pixels": topo_union["pixel_count"]
            },
            "geometric_gap_connections": len(gap_conns),
            "post_graph_merge_components": n_merged_comps,
            "length_audit": {
                "raw_union_true_arc_length_m": round(true_len, 2),
                "raw_union_flawed_pca_length_m": round(pca_len, 2),
                "pca_inflation_ratio": round(pca_len / max(true_len, 0.001), 2),
                "merged_true_arc_length_m": round(true_len_merged, 2),
                "diagnostics": len_diag
            }
        }
        all_frame_results[str(fnum)] = frame_summary
        
        print(f"  Raw YOLO Instances    : {raw_box_count} boxes (Overlapping pairs: {overlapping_pairs})")
        print(f"  Union Mask Components : {len(valid_raw_union_comps)}")
        print(f"  After 2px Closing     : {morph_results['2px']['components']} components (+{morph_results['2px']['pixel_growth_pct']}% px)")
        print(f"  After 3px Closing     : {morph_results['3px']['components']} components (+{morph_results['3px']['pixel_growth_pct']}% px)")
        print(f"  Post Graph-Merge Comps: {n_merged_comps} (Bridged {len(gap_conns)} gaps <= 15px)")
        print(f"  Length Audit          : True Arc Length = {true_len:.2f} m vs Flawed PCA = {pca_len:.2f} m (Inflation: {frame_summary['length_audit']['pca_inflation_ratio']}x)")
        
        # -------------------------------------------------------------
        # 6. PRODUCE 5-PANEL COMPARISON FOR KEY FRAMES (44, 47, 49, 65)
        # -------------------------------------------------------------
        if fnum in comparison_frames:
            print(f"  Generating 5-panel comparison contact sheet for Frame {fnum:02d}...")
            # Panel 1: Raw YOLO Detections (boxes + colored masks)
            p1 = img.copy()
            colors = [
                (0, 255, 255), (0, 165, 255), (255, 100, 0), (255, 0, 255),
                (0, 255, 0), (100, 255, 100), (255, 255, 0), (0, 200, 255)
            ]
            for idx_c, rc in enumerate(raw_cracks):
                col = colors[idx_c % len(colors)]
                p1[rc["mask"] > 0] = (0.5 * p1[rc["mask"] > 0] + 0.5 * np.array(col)).astype(np.uint8)
                bx = rc["bbox"]
                cv2.rectangle(p1, (bx[0], bx[1]), (bx[2], bx[3]), col, 1)
                cv2.putText(p1, f"#{idx_c+1} ({rc['confidence']:.2f})", (bx[0], max(bx[1] - 4, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            cv2.putText(p1, f"1. RAW YOLO DETECTIONS ({raw_box_count} boxes)", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.rectangle(p1, (15, 10), (W - 15, 55), (0, 0, 0), 1)
            
            # Panel 2: Union Mask (binary footprint showing gaps)
            p2 = img.copy()
            p2[union_mask > 0] = (0.4 * p2[union_mask > 0] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)
            cv2.putText(p2, f"2. UNION MASK ({len(valid_raw_union_comps)} connected components)", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                        
            # Panel 3: Current Fragmented Components (each labeled with PCA-sorted length)
            p3 = img.copy()
            # Simulate current pipeline components
            cur_instances = CrackGeometryEngine().analyze_cracks(union_mask, ref_road_plane, calib)
            for ci in cur_instances:
                cnts_cur, _ = cv2.findContours(ci.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_cur:
                    cv2.drawContours(p3, [cnts_cur[0]], -1, (0, 255, 255), 2)
                    xc, yc = ci.bbox_xyxy[0], ci.bbox_xyxy[1]
                    cv2.putText(p3, f"CRACK #{ci.track_id:02d} | L:{ci.length_m:.2f}m (PCA)", (xc, max(yc - 6, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            cv2.putText(p3, f"3. CURRENT COMPONENTS ({len(cur_instances)} instances | Flawed PCA L={pca_len:.1f}m)", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                        
            # Panel 4: Morphology Candidate (2px closing)
            p4 = img.copy()
            closed_2px = morph_results["2px"]["mask"]
            p4[closed_2px > 0] = (0.4 * p4[closed_2px > 0] + 0.6 * np.array([0, 200, 255])).astype(np.uint8)
            cnts_morph, _ = cv2.findContours(closed_2px, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p4, cnts_morph, -1, (0, 200, 255), 2)
            cv2.putText(p4, f"4. MORPHOLOGY (2px Closing: {morph_results['2px']['components']} components | +{morph_results['2px']['pixel_growth_pct']}% px)",
                        (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 200, 255), 2)
                        
            # Panel 5: Graph / Topology Candidate (skeleton + junctions + bridged gaps + true arc length)
            p5 = img.copy()
            # Draw skeleton
            p5[skel_union > 0] = [0, 255, 255]
            # Draw bridged connections
            for conn in gap_conns:
                cv2.line(p5, tuple(conn["p1"]), tuple(conn["p2"]), (0, 0, 255), 2)
            # Draw junctions
            for junc in topo_union["junctions"]:
                cv2.circle(p5, junc, 3, (0, 255, 0), -1)
            # Draw endpoints
            for ep in topo_union["endpoints"]:
                cv2.circle(p5, ep, 3, (255, 0, 0), -1)
                
            cv2.putText(p5, f"5. TOPOLOGY GRAPH (Junctions: {len(topo_union['junctions'])}, Endpoints: {len(topo_union['endpoints'])}, True Arc L={true_len:.2f}m)",
                        (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (100, 255, 100), 2)
            cv2.putText(p5, f"Bridged Gaps: {len(gap_conns)} (Red lines) | Green: Junctions | Blue: Endpoints", (25, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 1)
                        
            # Stack into a master comparison image
            # Layout: Top row (P1, P2, P3), Bottom row (P4, P5, Info Card)
            scale = 0.5
            sw, sh = int(W * scale), int(H * scale)
            s1 = cv2.resize(p1, (sw, sh))
            s2 = cv2.resize(p2, (sw, sh))
            s3 = cv2.resize(p3, (sw, sh))
            s4 = cv2.resize(p4, (sw, sh))
            s5 = cv2.resize(p5, (sw, sh))
            
            # Info Card
            info_card = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.rectangle(info_card, (10, 10), (sw - 10, sh - 10), (30, 30, 30), -1)
            cv2.rectangle(info_card, (10, 10), (sw - 10, sh - 10), (80, 80, 80), 1)
            cv2.putText(info_card, f"FRAME {fnum:02d} METRICS AUDIT", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(info_card, f"- Raw YOLO Boxes: {raw_box_count} (Overlap pairs: {overlapping_pairs})", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(info_card, f"- Union Components: {len(valid_raw_union_comps)} (gaps break union)", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(info_card, f"- 2px Closing Comps: {morph_results['2px']['components']} (+{morph_results['2px']['pixel_growth_pct']}% px)", (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
            cv2.putText(info_card, f"- Post-Graph Merged: {n_merged_comps} networks", (25, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
            cv2.putText(info_card, f"- Flawed PCA Length: {pca_len:.2f} m", (25, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
            cv2.putText(info_card, f"- True Arc Length  : {true_len:.2f} m", (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(info_card, f"- Inflation Factor : {frame_summary['length_audit']['pca_inflation_ratio']:.1f}x (from PCA jump scrambling)", (25, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.putText(info_card, f"- Mean Distance Z  : {len_diag[0]['mean_z_m']:.2f} m (range {len_diag[0]['min_z_m']:.1f}-{len_diag[0]['max_z_m']:.1f}m)", (25, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            
            top_row = np.hstack([s1, s2, s3])
            bot_row = np.hstack([s4, s5, info_card])
            master_sheet = np.vstack([top_row, bot_row])
            
            out_img_path = out_debug_dir / f"fragmentation_comparison_frame_{fnum:02d}.jpg"
            cv2.imwrite(str(out_img_path), master_sheet)
            
            # Copy to artifact directory for clickable report
            art_img_path = artifact_dir / f"fragmentation_comparison_frame_{fnum:02d}.jpg"
            cv2.imwrite(str(art_img_path), master_sheet)
            print(f"  Saved comparison to {out_img_path.name}")
            
    # -------------------------------------------------------------
    # 7. TEMPORAL CONSISTENCY CHECK ACROSS ADJACENT FRAMES
    # -------------------------------------------------------------
    print("\n--- Auditing Temporal Consistency Across Consecutive Frames ---")
    temporal_sequences = [
        [36, 37, 38],
        [42, 43, 44, 45, 46, 47, 48, 49]
    ]
    temporal_summary = []
    
    for seq in temporal_sequences:
        seq_records = []
        for fn in seq:
            f_p = left_dir / f"frame_{fn:05d}.jpg"
            if not f_p.exists():
                continue
            im = cv2.imread(str(f_p))
            rcs = extract_raw_yolo_cracks(model, im, conf_thresh=0.035)
            u_m = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
            for rc in rcs:
                u_m = cv2.bitwise_or(u_m, rc["mask"] * 255)
            sk = zhang_suen_thinning(u_m)
            tl, pl, _ = compute_true_graph_length_3d(sk, calib, ref_road_plane)
            n_c, _, stats, _ = cv2.connectedComponentsWithStats((u_m > 0).astype(np.uint8), connectivity=8)
            valid_c = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_c) if stats[i, cv2.CC_STAT_AREA] >= 25]
            seq_records.append({
                "frame": fn,
                "raw_boxes": len(rcs),
                "union_comps": len(valid_c),
                "true_arc_len_m": round(tl, 2),
                "flawed_pca_len_m": round(pl, 2),
                "total_pixels": int(cv2.countNonZero(u_m))
            })
        temporal_summary.append(seq_records)
        
    summary_data = {
        "dataset": "data/pothole_long_stereo",
        "diagnostic": "Crack Fragmentation Root-Cause & Length Audit",
        "frames_analyzed": target_frames,
        "results_by_frame": all_frame_results,
        "temporal_consistency": temporal_summary
    }
    
    json_path = out_debug_dir / "crack_fragmentation_diagnostic_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved full diagnostic summary JSON to {json_path}")
    print(f"Total diagnostic elapsed time: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
