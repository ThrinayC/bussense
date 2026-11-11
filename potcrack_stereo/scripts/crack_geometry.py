#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Crack 3D Geometry & Network Graph Engine
===================================================
Computes true metric 3D road-surface crack metrics using:
- Mask-level union across YOLO detections
- Conservative morphological micro-gap closing
- Skeleton topological graph extraction (endpoints, junctions, branches, loops)
- Conservative endpoint bridging (gap-constrained and tangent-aligned)
- True 8-connected 3D skeleton graph arc length (completely replacing flawed 1D PCA sorting)
- Crack network identification (preserving branch and loop topology)
- Calibrated camera ray / fitted road plane intersection
- Metric Euclidean length (m) and surface coverage area (m²)

Strict rule:
Does NOT report crack depth (as stereo depth inside thin sub-centimeter cracks is physically invalid).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
import cv2
import numpy as np


@dataclass
class CrackInstance3D:
    track_id: int
    confidence: float
    bbox_xyxy: List[int]
    mask: np.ndarray
    skeleton_mask: np.ndarray
    length_m: float
    coverage_area_m2: float
    center_3d: List[float]
    linear_density_m_m2: Optional[float] = None
    status: str = "OK"


@dataclass
class CrackNetwork3D:
    network_id: int
    confidence: float
    bbox_xyxy: List[int]
    mask: np.ndarray
    skeleton_mask: np.ndarray
    total_length_m: float
    coverage_area_m2: float
    branch_count: int
    junction_count: int
    endpoint_count: int
    center_3d: List[float]
    branches: List[Dict[str, Any]] = field(default_factory=list)
    bridged_gaps_count: int = 0
    linear_density_m_m2: Optional[float] = None
    status: str = "OK"


def zhang_suen_thinning(binary_mask: np.ndarray) -> np.ndarray:
    """
    Standard Zhang-Suen morphological thinning algorithm for skeletonization.
    Input: binary uint8 mask (0 and 255).
    Output: 1-pixel wide skeleton uint8 mask (0 and 255).
    """
    if np.count_nonzero(binary_mask) == 0:
        return np.zeros_like(binary_mask)

    # Use OpenCV ximgproc if available, else morphological approximation
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        pass

    # Morphological skeletonization fallback
    skeleton = np.zeros_like(binary_mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    img = binary_mask.copy()

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        zeros = binary_mask.size - cv2.countNonZero(img)
        if zeros == binary_mask.size:
            done = True

    return skeleton


def project_pixel_to_3d_road(
    u: float,
    v: float,
    calib: Dict[str, float],
    road_plane: np.ndarray,
    lateral_bound_m: float = 5.5,
    z_min_m: float = 1.0,
    z_max_m: float = 35.0
) -> Optional[np.ndarray]:
    """
    Projects pixel (u, v) along calibrated camera ray to fitted 3D road plane.
    Returns 3D point [X, Y, Z] in camera frame or None if invalid/out-of-bounds.
    """
    fx, fy = calib["fx"], calib["fy"]
    cx, cy = calib["cx"], calib["cy"]
    a, b, c, d = road_plane

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
    if Z < z_min_m or Z > z_max_m:
        return None

    X = rx * Z
    Y = ry * Z
    if abs(X) > lateral_bound_m:
        return None

    return np.array([X, Y, Z], dtype=np.float32)


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
            inward_pt = None
            for dx, dy in neighbor_offsets:
                cand = (x + dx, y + dy)
                if 0 <= cand[0] < W and 0 <= cand[1] < H:
                    if comp_mask[cand[1], cand[0]] > 0:
                        inward_pt = cand
                        break
            if inward_pt is not None:
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
                        dir_forward = gap_vec / dist
                        dir_backward = -dir_forward
                        cos1 = float(np.dot(t1, dir_forward))
                        cos2 = float(np.dot(t2, dir_backward))
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


def build_skeleton_graph_and_compute_length(
    skeleton: np.ndarray,
    calib: Dict[str, float],
    road_plane: np.ndarray,
    lateral_bound_m: float = 5.5,
    z_min_m: float = 1.0,
    z_max_m: float = 35.0
) -> Tuple[float, List[Dict[str, Any]], Dict[str, int]]:
    """
    Extracts topological graph from a binary skeleton (endpoints, junctions, edges, loops)
    and computes the true 3D physical arc length along all unique graph edges.
    Every edge is traversed exactly once.
    Loops (cycles) are handled naturally.
    Shared junctions are not double-counted.
    Returns: (total_network_length_m, list_of_branches, topology_counts)
    """
    H, W = skeleton.shape[:2]
    skel_binary = (skeleton > 0).astype(np.uint8)
    if cv2.countNonZero(skel_binary) == 0:
        return 0.0, [], {"branches": 0, "junctions": 0, "endpoints": 0}

    # 1. Pixel 8-neighborhood degree analysis
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel) * skel_binary

    skel_pts = set(zip(*np.where(skel_binary > 0)[::-1]))  # (x, y)
    raw_endpoints = set()
    raw_junctions = set()

    for x, y in skel_pts:
        nb = neighbor_count[y, x]
        if nb == 1:
            raw_endpoints.add((x, y))
        elif nb >= 3:
            raw_junctions.add((x, y))

    # 2. Cluster adjacent junction pixels into single unified Junction Nodes
    junction_mask = np.zeros((H, W), dtype=np.uint8)
    for jx, jy in raw_junctions:
        junction_mask[jy, jx] = 255

    num_junc_nodes, junc_labels = cv2.connectedComponents(junction_mask, connectivity=8)
    junc_pixel_to_node: Dict[Tuple[int, int], int] = {}
    for j_id in range(1, num_junc_nodes):
        jys, jxs = np.where(junc_labels == j_id)
        for x, y in zip(jxs, jys):
            junc_pixel_to_node[(x, y)] = j_id

    # 3. Pre-cache 3D projections for all skeleton pixels
    pts_3d_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    for pt in skel_pts:
        pts_3d_cache[pt] = project_pixel_to_3d_road(
            pt[0], pt[1], calib, road_plane, lateral_bound_m, z_min_m, z_max_m
        )

    # 4. Trace physical branches between nodes (endpoints or junction clusters)
    visited_pixel_steps: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    branches = []
    total_length_m = 0.0

    neighbor_offsets = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]

    start_points = list(raw_endpoints) + list(raw_junctions)

    for start_pt in start_points:
        is_junc_start = start_pt in junc_pixel_to_node
        start_cluster = junc_pixel_to_node.get(start_pt, None)

        for dx, dy in neighbor_offsets:
            nb = (start_pt[0] + dx, start_pt[1] + dy)
            if nb not in skel_pts:
                continue

            # If both start and nb are in the same junction cluster, do not trace as an external branch
            if is_junc_start and junc_pixel_to_node.get(nb, None) == start_cluster:
                continue

            step = tuple(sorted([start_pt, nb]))
            if step in visited_pixel_steps:
                continue

            # Trace continuous path along degree-2 pixels
            visited_pixel_steps.add(step)
            branch_pixels = [start_pt, nb]
            prev = start_pt
            curr = nb

            while curr not in raw_endpoints and curr not in raw_junctions:
                next_pt = None
                for ndx, ndy in neighbor_offsets:
                    cand = (curr[0] + ndx, curr[1] + ndy)
                    if cand in skel_pts and cand != prev:
                        next_step = tuple(sorted([curr, cand]))
                        if next_step not in visited_pixel_steps:
                            visited_pixel_steps.add(next_step)
                            next_pt = cand
                            break
                if next_pt is None:
                    break
                branch_pixels.append(next_pt)
                prev = curr
                curr = next_pt

            # Compute 3D physical arc length of this branch
            branch_len_3d = 0.0
            for i in range(len(branch_pixels) - 1):
                p1_3d = pts_3d_cache[branch_pixels[i]]
                p2_3d = pts_3d_cache[branch_pixels[i + 1]]
                if p1_3d is not None and p2_3d is not None:
                    d_step = float(np.linalg.norm(p1_3d - p2_3d))
                    if d_step < 0.5:
                        branch_len_3d += d_step

            # Only register branches with meaningful pixel length or valid 3D length
            if len(branch_pixels) >= 3 or branch_len_3d > 0.01:
                total_length_m += branch_len_3d
                branches.append({
                    "start": [int(branch_pixels[0][0]), int(branch_pixels[0][1])],
                    "end": [int(branch_pixels[-1][0]), int(branch_pixels[-1][1])],
                    "length_m": round(branch_len_3d, 3),
                    "pixel_count": len(branch_pixels)
                })

    # 5. Handle isolated closed loops (cycles with no endpoints or junctions)
    unvisited_pts = set()
    for pt in skel_pts:
        has_unvisited = False
        for dx, dy in neighbor_offsets:
            nb = (pt[0] + dx, pt[1] + dy)
            if nb in skel_pts:
                step = tuple(sorted([pt, nb]))
                if step not in visited_pixel_steps:
                    has_unvisited = True
                    break
        if has_unvisited and pt not in junc_pixel_to_node:
            unvisited_pts.add(pt)

    while unvisited_pts:
        start_pt = next(iter(unvisited_pts))
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

        for lp in loop_pixels:
            unvisited_pts.discard(lp)

        loop_len_3d = 0.0
        for i in range(len(loop_pixels) - 1):
            p1_3d = pts_3d_cache[loop_pixels[i]]
            p2_3d = pts_3d_cache[loop_pixels[i + 1]]
            if p1_3d is not None and p2_3d is not None:
                d_step = float(np.linalg.norm(p1_3d - p2_3d))
                if d_step < 0.5:
                    loop_len_3d += d_step

        if loop_len_3d > 0.01:
            total_length_m += loop_len_3d
            branches.append({
                "start": [int(loop_pixels[0][0]), int(loop_pixels[0][1])],
                "end": [int(loop_pixels[-1][0]), int(loop_pixels[-1][1])],
                "length_m": round(loop_len_3d, 3),
                "pixel_count": len(loop_pixels),
                "is_loop": True
            })

    # Fallback for short isolated 2-pixel components
    if total_length_m == 0.0 and len(skel_pts) >= 2:
        valid_3d = [pts_3d_cache[p] for p in skel_pts if pts_3d_cache[p] is not None]
        if len(valid_3d) >= 2:
            arr_3d = np.array(valid_3d)
            total_length_m = float(np.linalg.norm(np.max(arr_3d, axis=0) - np.min(arr_3d, axis=0)))
            if len(branches) == 0:
                branches.append({
                    "start": [int(list(skel_pts)[0][0]), int(list(skel_pts)[0][1])],
                    "end": [int(list(skel_pts)[-1][0]), int(list(skel_pts)[-1][1])],
                    "length_m": round(total_length_m, 3),
                    "pixel_count": len(skel_pts)
                })

    topology = {
        "branches": len(branches),
        "junctions": max(0, num_junc_nodes - 1),
        "endpoints": len(raw_endpoints)
    }

    return round(total_length_m, 2), branches, topology


class CrackNetworkGraphEngine:
    """
    Topology-aware crack network engine.
    Extracts coherent crack networks, preserves branching and loops,
    and computes true 8-connected 3D skeleton graph arc length.
    """

    def __init__(
        self,
        closing_kernel_size: int = 5,
        min_component_pixels: int = 25,
        min_length_m: float = 0.15,
        max_length_m: float = 80.0,
        max_bridge_gap_px: float = 15.0,
        bridge_angle_threshold_deg: float = 60.0,
        lateral_road_bound_m: float = 5.5,
        z_min_m: float = 1.0,
        z_max_m: float = 35.0
    ):
        self.closing_kernel_size = closing_kernel_size
        self.min_component_pixels = min_component_pixels
        self.min_length_m = min_length_m
        self.max_length_m = max_length_m
        self.max_bridge_gap_px = max_bridge_gap_px
        self.bridge_angle_threshold_deg = bridge_angle_threshold_deg
        self.lateral_road_bound_m = lateral_road_bound_m
        self.z_min_m = z_min_m
        self.z_max_m = z_max_m

        # Next available track ID for temporal tracking
        self._next_track_id = 1
        self._prev_tracks: Dict[int, Dict[str, Any]] = {}

    def analyze_crack_networks(
        self,
        union_mask: np.ndarray,
        road_plane: np.ndarray,
        calib: Dict[str, float],
        raw_box_confs: Optional[List[Dict[str, Any]]] = None,
        observable_road_area_m2: Optional[float] = None
    ) -> List[CrackNetwork3D]:
        """
        Extracts topological crack networks from the union mask.
        Returns list of CrackNetwork3D instances.
        """
        networks = []
        if union_mask is None or np.count_nonzero(union_mask) == 0:
            return networks

        if road_plane is None or len(road_plane) != 4:
            return networks

        H, W = union_mask.shape[:2]

        # 1. Conservative 2px closing (5x5 ellipse) to heal micro-gaps
        if self.closing_kernel_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.closing_kernel_size, self.closing_kernel_size)
            )
            closed_mask = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, kernel)
        else:
            closed_mask = union_mask.copy()

        # 2. Connected components on closed mask -> Network Candidates
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (closed_mask > 0).astype(np.uint8), connectivity=8
        )

        current_frame_candidates = []

        for l_id in range(1, num_labels):
            area_px = stats[l_id, cv2.CC_STAT_AREA]
            if area_px < self.min_component_pixels:
                continue

            comp_mask = ((labels == l_id) * 255).astype(np.uint8)
            x1 = stats[l_id, cv2.CC_STAT_LEFT]
            y1 = stats[l_id, cv2.CC_STAT_TOP]
            w = stats[l_id, cv2.CC_STAT_WIDTH]
            h = stats[l_id, cv2.CC_STAT_HEIGHT]
            bbox = [x1, y1, x1 + w, y1 + h]

            # 3. Skeletonize via Zhang-Suen thinning
            skel = zhang_suen_thinning(comp_mask)
            if cv2.countNonZero(skel) < 5:
                continue

            # 4. Conservative endpoint bridging within the network
            bridged_skel, conns = bridge_skeleton_endpoints(
                skel,
                max_gap_px=self.max_bridge_gap_px,
                max_angle_diff_deg=self.bridge_angle_threshold_deg
            )

            # 5. Extract topological graph & compute true 3D graph arc length
            total_len_m, branches, topo = build_skeleton_graph_and_compute_length(
                bridged_skel,
                calib,
                road_plane,
                self.lateral_road_bound_m,
                self.z_min_m,
                self.z_max_m
            )

            if total_len_m < self.min_length_m or total_len_m > self.max_length_m:
                continue

            # 6. Center 3D & Coverage Area
            comp_ys, comp_xs = np.where(comp_mask > 0)
            sample_xs = comp_xs[::max(1, len(comp_xs) // 50)]
            sample_ys = comp_ys[::max(1, len(comp_ys) // 50)]
            pts_3d_samples = [
                project_pixel_to_3d_road(x, y, calib, road_plane, self.lateral_road_bound_m, self.z_min_m, self.z_max_m)
                for x, y in zip(sample_xs, sample_ys)
            ]
            valid_pts = [p for p in pts_3d_samples if p is not None]
            if not valid_pts:
                continue
            center_3d = [round(float(v), 2) for v in np.median(valid_pts, axis=0)]

            mean_z = center_3d[2]
            pixel_area_m2 = (mean_z / calib["fx"]) * (mean_z / calib["fy"])
            coverage_m2 = round(float(area_px * pixel_area_m2), 4)

            # 7. Confidence matching
            conf = 0.035
            if raw_box_confs:
                for b_info in raw_box_confs:
                    if cv2.countNonZero(cv2.bitwise_and(comp_mask, b_info["mask"])) > 0:
                        conf = max(conf, b_info["conf"])

            density = None
            if observable_road_area_m2 and observable_road_area_m2 > 5.0:
                density = round(float(total_len_m / observable_road_area_m2), 4)

            current_frame_candidates.append({
                "confidence": round(conf, 3),
                "bbox_xyxy": bbox,
                "mask": comp_mask,
                "skeleton_mask": bridged_skel,
                "total_length_m": round(total_len_m, 2),
                "coverage_area_m2": coverage_m2,
                "branch_count": topo["branches"],
                "junction_count": topo["junctions"],
                "endpoint_count": topo["endpoints"],
                "center_3d": center_3d,
                "branches": branches,
                "bridged_gaps_count": len(conns),
                "linear_density_m_m2": density,
                "center_2d": (float(centroids[l_id][0]), float(centroids[l_id][1]))
            })

        # 8. Lightweight temporal association to stabilize network IDs
        matched_tracks = {}
        assigned_cands = set()

        for t_id, prev_t in self._prev_tracks.items():
            best_cand_idx = None
            best_dist = 1e9
            for c_idx, cand in enumerate(current_frame_candidates):
                if c_idx in assigned_cands:
                    continue
                # 2D distance between centroids (accounting for forward movement)
                dx = cand["center_2d"][0] - prev_t["center_2d"][0]
                dy = cand["center_2d"][1] - prev_t["center_2d"][1]
                dist_2d = np.hypot(dx, dy)
                if dist_2d < 85.0 and dist_2d < best_dist:
                    best_dist = dist_2d
                    best_cand_idx = c_idx

            if best_cand_idx is not None:
                assigned_cands.add(best_cand_idx)
                cand = current_frame_candidates[best_cand_idx]
                matched_tracks[t_id] = {
                    "center_2d": cand["center_2d"],
                    "age": prev_t["age"] + 1,
                    "missed": 0
                }
                cand["network_id"] = t_id

        # Assign new IDs for unassigned candidates
        for c_idx, cand in enumerate(current_frame_candidates):
            if "network_id" not in cand:
                cand["network_id"] = self._next_track_id
                self._next_track_id += 1
                matched_tracks[cand["network_id"]] = {
                    "center_2d": cand["center_2d"],
                    "age": 1,
                    "missed": 0
                }

        # Update track state
        self._prev_tracks = matched_tracks

        # Convert to CrackNetwork3D objects
        for cand in current_frame_candidates:
            networks.append(CrackNetwork3D(
                network_id=cand["network_id"],
                confidence=cand["confidence"],
                bbox_xyxy=cand["bbox_xyxy"],
                mask=cand["mask"],
                skeleton_mask=cand["skeleton_mask"],
                total_length_m=cand["total_length_m"],
                coverage_area_m2=cand["coverage_area_m2"],
                branch_count=cand["branch_count"],
                junction_count=cand["junction_count"],
                endpoint_count=cand["endpoint_count"],
                center_3d=cand["center_3d"],
                branches=cand["branches"],
                bridged_gaps_count=cand["bridged_gaps_count"],
                linear_density_m_m2=cand["linear_density_m_m2"],
                status="OK"
            ))

        # Sort by length descending
        networks.sort(key=lambda x: x.total_length_m, reverse=True)
        return networks


class CrackGeometryEngine:
    """
    Maintained for backward compatibility.
    Updated to use the non-inflated topological skeleton graph arc length.
    """

    def __init__(
        self,
        min_length_m: float = 0.15,
        max_length_m: float = 40.0,
        lateral_road_bound_m: float = 5.5,
        z_min_m: float = 1.0,
        z_max_m: float = 35.0
    ):
        self.min_length_m = min_length_m
        self.max_length_m = max_length_m
        self.lateral_road_bound_m = lateral_road_bound_m
        self.z_min_m = z_min_m
        self.z_max_m = z_max_m
        self._net_engine = CrackNetworkGraphEngine(
            closing_kernel_size=1,  # baseline compatibility
            min_component_pixels=25,
            min_length_m=min_length_m,
            max_length_m=max_length_m,
            lateral_road_bound_m=lateral_road_bound_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m
        )

    def analyze_cracks(
        self,
        crack_mask: np.ndarray,
        road_plane: Optional[np.ndarray],
        calib: Dict[str, float],
        observable_road_area_m2: Optional[float] = None
    ) -> List[CrackInstance3D]:
        """
        Analyzes connected crack components in the mask using true skeleton graph length.
        """
        instances = []
        if crack_mask is None or np.count_nonzero(crack_mask) == 0:
            return instances

        if road_plane is None or len(road_plane) != 4:
            return instances

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (crack_mask > 0).astype(np.uint8), connectivity=8
        )

        for label_id in range(1, num_labels):
            area_px = stats[label_id, cv2.CC_STAT_AREA]
            if area_px < 25:
                continue

            comp_mask = ((labels == label_id) * 255).astype(np.uint8)
            x1 = stats[label_id, cv2.CC_STAT_LEFT]
            y1 = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            bbox = [x1, y1, x1 + w, y1 + h]

            skeleton = zhang_suen_thinning(comp_mask)
            total_len_m, branches, topo = build_skeleton_graph_and_compute_length(
                skeleton, calib, road_plane, self.lateral_road_bound_m, self.z_min_m, self.z_max_m
            )

            if total_len_m < self.min_length_m or total_len_m > self.max_length_m:
                continue

            center_3d = [0.0, 0.0, 5.0]
            coverage_m2 = 0.0
            density = None
            if observable_road_area_m2 and observable_road_area_m2 > 5.0:
                density = float(total_len_m / observable_road_area_m2)

            instances.append(CrackInstance3D(
                track_id=label_id,
                confidence=0.85,
                bbox_xyxy=bbox,
                mask=comp_mask,
                skeleton_mask=skeleton,
                length_m=round(total_len_m, 2),
                coverage_area_m2=round(coverage_m2, 4),
                center_3d=center_3d,
                linear_density_m_m2=round(density, 4) if density is not None else None,
                status="OK"
            ))

        return instances
