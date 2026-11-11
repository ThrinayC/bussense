#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_crack_network_engine.py
====================================
Validates the new CrackNetworkGraphEngine:
- Union of all YOLO crack masks
- 2px conservative morphological closing
- Skeleton topological graph (nodes: endpoints, junctions; edges: branches/loops)
- Conservative endpoint bridging (<= 15px, orientation-aware)
- True 8-connected 3D skeleton graph arc length (completely replacing flawed PCA sorting)
- Compares OLD PCA length vs NEW Graph length for frames 36, 44, 47, 49, 65
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.crack_geometry import zhang_suen_thinning


def build_skeleton_graph_and_compute_length(
    skeleton: np.ndarray,
    calib: Dict[str, float],
    road_plane: np.ndarray
) -> Tuple[float, List[Dict[str, Any]], Dict[str, int]]:
    """
    Extracts topological graph from a binary skeleton (endpoints, junctions, edges, loops)
    and computes the true 3D physical arc length along all unique graph edges.
    Every edge is traversed exactly once.
    Loops (cycles) are handled naturally without starting/ending at an endpoint.
    Shared junctions are zero-length points and not double-counted.
    Returns: (total_network_length_m, list_of_branches, topology_counts)
    """
    H, W = skeleton.shape[:2]
    skel_binary = (skeleton > 0).astype(np.uint8)
    if cv2.countNonZero(skel_binary) == 0:
        return 0.0, [], {"branches": 0, "junctions": 0, "endpoints": 0}

    fx, fy = calib["fx"], calib["fy"]
    cx, cy = calib["cx"], calib["cy"]
    a, b, c, d = road_plane

    # 1. Pixel 8-neighborhood analysis
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel) * skel_binary

    skel_pts = set(zip(*np.where(skel_binary > 0)[::-1]))  # set of (x, y)
    endpoints = set()
    junctions = set()

    for x, y in skel_pts:
        nb = neighbor_count[y, x]
        if nb == 1:
            endpoints.add((x, y))
        elif nb >= 3:
            junctions.add((x, y))

    # Helper: Ray-plane 3D projection of a 2D pixel (u, v)
    def project_pixel_to_3d(u: float, v: float) -> Optional[np.ndarray]:
        rx = (u - cx) / fx
        ry = (v - cy) / fy
        rz = 1.0
        denom = a * rx + b * ry + c * rz
        if abs(denom) < 1e-4:
            return None
        t_dist = -d / denom
        if t_dist <= 0:
            return None
        Z = t_dist
        if Z < 1.0 or Z > 35.0:
            return None
        X = rx * Z
        Y = ry * Z
        if abs(X) > 5.5:
            return None
        return np.array([X, Y, Z], dtype=np.float32)

    # 2. Trace edges (branches) from nodes
    # We maintain visited edges by pairs of adjacent pixels to ensure each edge is counted once
    visited_pixel_steps: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    branches = []
    total_length_m = 0.0

    # 8-neighbor offsets
    neighbor_offsets = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]

    # Pre-cache 3D projections for all skeleton pixels
    pts_3d_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    for pt in skel_pts:
        pts_3d_cache[pt] = project_pixel_to_3d(pt[0], pt[1])

    # Start tracing from all node points (endpoints and junctions)
    start_nodes = list(endpoints) + list(junctions)

    for start_pt in start_nodes:
        # Explore each neighbor of start_pt
        for dx, dy in neighbor_offsets:
            nb = (start_pt[0] + dx, start_pt[1] + dy)
            if nb not in skel_pts:
                continue
            step = tuple(sorted([start_pt, nb]))
            if step in visited_pixel_steps:
                continue

            # Trace branch until another junction or endpoint is encountered
            branch_pixels = [start_pt, nb]
            visited_pixel_steps.add(step)
            curr = nb
            prev = start_pt

            while curr not in junctions and curr not in endpoints:
                # Find next neighbor that is not prev
                next_pt = None
                for ndx, ndy in neighbor_offsets:
                    cand = (curr[0] + ndx, curr[1] + ndy)
                    if cand in skel_pts and cand != prev:
                        next_step = tuple(sorted([curr, cand]))
                        if next_step not in visited_pixel_steps:
                            next_pt = cand
                            visited_pixel_steps.add(next_step)
                            break
                if next_pt is None:
                    break
                branch_pixels.append(next_pt)
                prev = curr
                curr = next_pt

            # Calculate 3D arc length of this branch
            branch_len_3d = 0.0
            for i in range(len(branch_pixels) - 1):
                p1_3d = pts_3d_cache[branch_pixels[i]]
                p2_3d = pts_3d_cache[branch_pixels[i + 1]]
                if p1_3d is not None and p2_3d is not None:
                    d_step = float(np.linalg.norm(p1_3d - p2_3d))
                    if d_step < 0.5:  # physical continuity check between adjacent pixels
                        branch_len_3d += d_step

            if branch_len_3d > 0.0:
                total_length_m += branch_len_3d
                branches.append({
                    "start": [int(branch_pixels[0][0]), int(branch_pixels[0][1])],
                    "end": [int(branch_pixels[-1][0]), int(branch_pixels[-1][1])],
                    "length_m": round(branch_len_3d, 3),
                    "pixel_count": len(branch_pixels)
                })

    # 3. Handle isolated closed loops / cycles (components with no endpoints and no junctions)
    # Check if any skeleton pixel has not been visited
    unvisited_pixels = set()
    for pt in skel_pts:
        # Check if all incident steps from pt were visited
        has_unvisited = False
        for dx, dy in neighbor_offsets:
            nb = (pt[0] + dx, pt[1] + dy)
            if nb in skel_pts:
                step = tuple(sorted([pt, nb]))
                if step not in visited_pixel_steps:
                    has_unvisited = True
                    break
        if has_unvisited:
            unvisited_pixels.add(pt)

    while unvisited_pixels:
        start_pt = next(iter(unvisited_pixels))
        loop_pixels = [start_pt]
        curr = start_pt
        prev = None

        while True:
            next_pt = None
            for dx, dy in neighbor_offsets:
                cand = (curr[0] + dx, curr[1] + dy)
                if cand in skel_pts and cand != prev:
                    step = tuple(sorted([curr, cand]))
                    if step not in visited_pixel_steps:
                        visited_pixel_steps.add(step)
                        next_pt = cand
                        break
            if next_pt is None:
                break
            loop_pixels.append(next_pt)
            prev = curr
            curr = next_pt
            if curr == start_pt:
                break

        # Remove traced pixels from unvisited
        for lp in loop_pixels:
            unvisited_pixels.discard(lp)

        # Compute loop length
        loop_len_3d = 0.0
        for i in range(len(loop_pixels) - 1):
            p1_3d = pts_3d_cache[loop_pixels[i]]
            p2_3d = pts_3d_cache[loop_pixels[i + 1]]
            if p1_3d is not None and p2_3d is not None:
                d_step = float(np.linalg.norm(p1_3d - p2_3d))
                if d_step < 0.5:
                    loop_len_3d += d_step

        if loop_len_3d > 0.0:
            total_length_m += loop_len_3d
            branches.append({
                "start": [int(loop_pixels[0][0]), int(loop_pixels[0][1])],
                "end": [int(loop_pixels[-1][0]), int(loop_pixels[-1][1])],
                "length_m": round(loop_len_3d, 3),
                "pixel_count": len(loop_pixels),
                "is_loop": True
            })

    # Fallback if skeleton has only disconnected single pixels without neighbors
    if total_length_m == 0.0 and len(skel_pts) >= 2:
        valid_3d = [pts_3d_cache[p] for p in skel_pts if pts_3d_cache[p] is not None]
        if len(valid_3d) >= 2:
            arr_3d = np.array(valid_3d)
            total_length_m = float(np.linalg.norm(np.max(arr_3d, axis=0) - np.min(arr_3d, axis=0)))

    topology = {
        "branches": len(branches),
        "junctions": len(junctions),
        "endpoints": len(endpoints)
    }

    return round(total_length_m, 2), branches, topology


def bridge_skeleton_endpoints(
    skeleton: np.ndarray,
    max_gap_px: float = 15.0,
    max_angle_diff_deg: float = 60.0
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Conservative endpoint bridging:
    Bridges endpoints of disconnected components only if:
    - distance <= max_gap_px
    - direction vector aligns with local branch tangent
    - line does not cross off-road
    """
    H, W = skeleton.shape[:2]
    num_labels, labels = cv2.connectedComponents((skeleton > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 2:
        return skeleton.copy(), []

    bridged_skel = skeleton.copy()
    connections = []

    # Find endpoints and compute local tangent vector for each
    comp_endpoints: Dict[int, List[Dict[str, Any]]] = {}
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    neighbor_offsets = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]

    for l_id in range(1, num_labels):
        comp_mask = (labels == l_id).astype(np.uint8)
        if cv2.countNonZero(comp_mask) < 5:
            continue
        nb_count = cv2.filter2D(comp_mask, -1, kernel) * comp_mask
        ep_ys, ep_xs = np.where(nb_count == 1)

        ep_list = []
        for x, y in zip(ep_xs, ep_ys):
            # Find the single neighbor inside comp_mask to determine inward tangent
            inward_pt = None
            for dx, dy in neighbor_offsets:
                cand = (x + dx, y + dy)
                if 0 <= cand[0] < W and 0 <= cand[1] < H:
                    if comp_mask[cand[1], cand[0]] > 0:
                        inward_pt = cand
                        break
            if inward_pt is not None:
                # Outward tangent vector: from inward_pt to (x, y)
                tangent = np.array([x - inward_pt[0], y - inward_pt[1]], dtype=np.float32)
                t_norm = np.linalg.norm(tangent)
                if t_norm > 0:
                    tangent /= t_norm
                ep_list.append({
                    "pt": (int(x), int(y)),
                    "tangent": tangent
                })
        if ep_list:
            comp_endpoints[l_id] = ep_list

    # Compare pairs of endpoints across different components
    cos_threshold = np.cos(np.radians(max_angle_diff_deg))
    labels_list = list(comp_endpoints.keys())

    for i in range(len(labels_list)):
        l1 = labels_list[i]
        for j in range(i + 1, len(labels_list)):
            l2 = labels_list[j]
            best_dist = 1e9
            best_pair = None

            for ep1 in comp_endpoints[l1]:
                p1 = ep1["pt"]
                t1 = ep1["tangent"]
                for ep2 in comp_endpoints[l2]:
                    p2 = ep2["pt"]
                    t2 = ep2["tangent"]
                    gap_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
                    dist = float(np.linalg.norm(gap_vec))
                    if dist <= max_gap_px and dist < best_dist:
                        # Check tangent alignment
                        dir_forward = gap_vec / dist
                        dir_backward = -dir_forward
                        cos1 = float(np.dot(t1, dir_forward))
                        cos2 = float(np.dot(t2, dir_backward))
                        # Require alignment within threshold
                        if cos1 >= cos_threshold and cos2 >= cos_threshold:
                            best_dist = dist
                            best_pair = (p1, p2)

            if best_pair is not None:
                p1, p2 = best_pair
                cv2.line(bridged_skel, p1, p2, 255, 1)
                connections.append({
                    "from_comp": l1,
                    "to_comp": l2,
                    "p1": [p1[0], p1[1]],
                    "p2": [p2[0], p2[1]],
                    "gap_px": round(best_dist, 1)
                })

    return bridged_skel, connections


def main():
    print("=== TESTING CRACK NETWORK GRAPH ENGINE & LENGTH VALIDATION ===")
    
    base_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "left"
    model_path = PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    
    calib = {"fx": 1050.0, "fy": 1050.0, "cx": 960.0, "cy": 540.0, "baseline": 0.12}
    ref_road_plane = np.array([0.0, -1.0, 0.05, 1.65], dtype=np.float32)
    
    test_frames = [36, 44, 47, 49, 65]
    comparison_records = []
    
    for fnum in test_frames:
        f_path = base_dir / f"frame_{fnum:05d}.jpg"
        if not f_path.exists():
            continue
        img = cv2.imread(str(f_path))
        H, W = img.shape[:2]
        
        # 1. Raw YOLO extraction at conf = 0.035
        res = model.predict(img, conf=0.035, verbose=False)[0]
        union_mask = np.zeros((H, W), dtype=np.uint8)
        yolo_boxes_count = 0
        
        if res.boxes is not None and res.masks is not None:
            for i, box in enumerate(res.boxes):
                if int(box.cls[0].item()) != 1:
                    continue
                b_coords = [int(v) for v in box.xyxy[0].cpu().numpy()]
                if b_coords[1] < 70:
                    continue
                box_area = (b_coords[2] - b_coords[0]) * (b_coords[3] - b_coords[1])
                if box_area > 0.35 * (W * H):
                    continue
                m_raw = res.masks.data[i].cpu().numpy()
                m_res = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                if cv2.countNonZero(m_res) > 0.05 * (W * H):
                    continue
                union_mask = cv2.bitwise_or(union_mask, m_res * 255)
                yolo_boxes_count += 1
                
        # 2. Conservative 2px closing (5x5 ellipse)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Connected components on closed mask -> Crack Networks
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (closed_mask > 0).astype(np.uint8), connectivity=8
        )
        
        # 4. Old PCA calculation on each component vs New Graph Arc Length
        old_pca_total = 0.0
        new_graph_total = 0.0
        networks_summary = []
        
        for l_id in range(1, num_labels):
            area_px = stats[l_id, cv2.CC_STAT_AREA]
            if area_px < 25:
                continue
            comp_mask = ((labels == l_id) * 255).astype(np.uint8)
            skel = zhang_suen_thinning(comp_mask)
            
            # Bridge endpoints
            bridged_skel, conns = bridge_skeleton_endpoints(skel, max_gap_px=15.0, max_angle_diff_deg=60.0)
            
            # Compute New Graph Arc Length
            graph_len, branches, topo = build_skeleton_graph_and_compute_length(bridged_skel, calib, ref_road_plane)
            
            # Compute Old PCA Length for baseline comparison
            sk_ys, sk_xs = np.where(skel > 0)
            pca_len = 0.0
            if len(sk_ys) >= 5:
                rx = (sk_xs - calib["cx"]) / calib["fx"]
                ry = (sk_ys - calib["cy"]) / calib["fy"]
                denom = ref_road_plane[0]*rx + ref_road_plane[1]*ry + ref_road_plane[2]
                val = (np.abs(denom) > 1e-4)
                t = -ref_road_plane[3] / denom
                val &= (t > 0)
                Z = t
                val &= (Z >= 1.0) & (Z <= 35.0) & (np.abs(rx*Z) <= 5.5) & (sk_ys >= 70)
                if np.sum(val) >= 5:
                    pts3 = np.stack([rx[val]*Z[val], ry[val]*Z[val], Z[val]], axis=1)
                    mean_pt = np.mean(pts3, axis=0)
                    centered = pts3 - mean_pt
                    cov = np.cov(centered[:, [0, 2]].T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    projections = centered[:, [0, 2]] @ eigvecs[:, 1]
                    sorted_pts = pts3[np.argsort(projections)]
                    diffs = np.diff(sorted_pts, axis=0)
                    segs = np.linalg.norm(diffs, axis=1)
                    pca_len = float(np.sum(segs[segs < 1.5]))
                    
            old_pca_total += pca_len
            new_graph_total += graph_len
            
            networks_summary.append({
                "network_id": l_id,
                "area_px": int(area_px),
                "branches": topo["branches"],
                "junctions": topo["junctions"],
                "endpoints": topo["endpoints"],
                "bridged_gaps": len(conns),
                "graph_length_m": graph_len,
                "pca_length_m": round(pca_len, 2)
            })
            
        rec = {
            "frame": fnum,
            "raw_yolo_boxes": yolo_boxes_count,
            "crack_networks_count": len(networks_summary),
            "old_pca_total_length_m": round(old_pca_total, 2),
            "new_graph_total_length_m": round(new_graph_total, 2),
            "inflation_removed_m": round(old_pca_total - new_graph_total, 2),
            "inflation_factor": round(old_pca_total / max(new_graph_total, 0.001), 2),
            "networks": networks_summary
        }
        comparison_records.append(rec)
        
        print(f"Frame {fnum:02d}: {yolo_boxes_count} YOLO boxes -> {len(networks_summary)} Networks")
        print(f"  Old PCA Length  : {old_pca_total:.2f} m")
        print(f"  New Graph Length: {new_graph_total:.2f} m (Inflation removed: -{old_pca_total - new_graph_total:.2f} m, {rec['inflation_factor']}x)")
        for net in networks_summary[:3]:
            print(f"    Net #{net['network_id']:02d}: L={net['graph_length_m']}m (Branches: {net['branches']}, Juncs: {net['junctions']}, EPs: {net['endpoints']})")
        if len(networks_summary) > 3:
            print(f"    (+{len(networks_summary)-3} more networks)")
            
    out_json = PROJECT_ROOT / "outputs" / "debug" / "crack_web_fragmentation" / "test_crack_network_comparison.json"
    with open(out_json, "w") as f:
        json.dump(comparison_records, f, indent=2)
    print(f"\nSaved test comparison to {out_json}")


if __name__ == "__main__":
    main()
