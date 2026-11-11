#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Phase 9: KITTI Tracking Stereo Traffic Benchmark
Evaluates real RGB-only stereo traffic perception against transformed KITTI 3D tracklet GT.

Strict Separation:
- Prediction path uses ONLY left RGB (image_02) and right RGB (image_03).
- ZERO LiDAR, ZERO precomputed GT depth, ZERO monocular depth at inference.
- GT tracklets are loaded ONLY for evaluation metrics.
- P_cam2 = R_rect_02 * (R_velo_to_cam * P_velo + T_velo_to_cam)
"""

import sys
import os
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import cv2

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic.tracker import TrafficTracker
from traffic.spatial_analyzer import TrafficSpatialAnalyzer
from traffic.density_engine import TrafficDensityEngine
from scripts.stereo_depth import get_stereo_matcher

# Map YOLO classes to matching KITTI classes
YOLO_TO_KITTI_MAP = {
    "car": ["Car", "Van"],
    "truck": ["Truck"],
    "bus": ["Truck", "Van", "Tram"],
    "motorcycle": ["Cyclist"]
}


class KittiTrackletEvaluator:
    """Parses tracklet_labels.xml and transforms Velodyne 3D coordinates to Camera 2 coordinates."""

    def __init__(self, calib_dir: Path, tracklet_xml_path: Path):
        self.tracklet_xml_path = tracklet_xml_path
        self._load_calibration(calib_dir)
        self.tracklets_by_frame = self._parse_tracklets()

    def _load_calibration(self, calib_dir: Path):
        # 1. R_velo_to_cam and T_velo_to_cam
        v2c_file = calib_dir / "calib_velo_to_cam.txt"
        with open(v2c_file, "r") as f:
            lines = f.readlines()
        v2c_dict = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    v2c_dict[k.strip()] = [float(x) for x in v.strip().split()]
                except ValueError:
                    pass
        self.R_v2c = np.array(v2c_dict["R"]).reshape(3, 3)
        self.T_v2c = np.array(v2c_dict["T"]).reshape(3, 1)

        # 2. R_rect_02 from calib_cam_to_cam.txt
        c2c_file = calib_dir / "calib_cam_to_cam.txt"
        with open(c2c_file, "r") as f:
            lines = f.readlines()
        c2c_dict = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    c2c_dict[k.strip()] = [float(x) for x in v.strip().split()]
                except ValueError:
                    pass
        self.R_rect_02 = np.array(c2c_dict["R_rect_02"]).reshape(3, 3)

        # Intrinsics P_rect_02: fx = P[0,0], cx = P[0,2], cy = P[1,2]
        P_rect = np.array(c2c_dict["P_rect_02"]).reshape(3, 4)
        self.fx = float(P_rect[0, 0])
        self.fy = float(P_rect[1, 1])
        self.cx = float(P_rect[0, 2])
        self.cy = float(P_rect[1, 2])
        # Baseline = -P_rect_03[0,3] / fx or ~0.5327m
        if "P_rect_03" in c2c_dict:
            P3 = np.array(c2c_dict["P_rect_03"]).reshape(3, 4)
            self.baseline = float(abs(P3[0, 3]) / self.fx)
        else:
            self.baseline = 0.5327

    def _transform_velo_to_cam2(self, tx: float, ty: float, tz: float) -> np.ndarray:
        """P_cam2 = R_rect_02 * (R_velo_to_cam * P_velo + T_velo_to_cam)"""
        P_velo = np.array([[tx], [ty], [tz]], dtype=np.float64)
        P_cam2 = self.R_rect_02 @ (self.R_v2c @ P_velo + self.T_v2c)
        return P_cam2.flatten()

    def _parse_tracklets(self) -> Dict[int, List[Dict[str, Any]]]:
        if not self.tracklet_xml_path.exists():
            return {}

        tree = ET.parse(self.tracklet_xml_path)
        root = tree.getroot()
        by_frame = {}

        for item_idx, item in enumerate(root.iter("item")):
            obj_type = item.find("objectType")
            first_frame = item.find("first_frame")
            poses = item.find("poses")

            if obj_type is None or first_frame is None or poses is None:
                continue

            kitti_type = obj_type.text.strip()
            start_f = int(first_frame.text)
            pose_items = poses.findall("item")

            for f_offset, pose in enumerate(pose_items):
                f_idx = start_f + f_offset
                tx = float(pose.find("tx").text)
                ty = float(pose.find("ty").text)
                tz = float(pose.find("tz").text)

                pos_cam2 = self._transform_velo_to_cam2(tx, ty, tz)
                if pos_cam2[2] > 0.5:  # Only objects in front of camera
                    if f_idx not in by_frame:
                        by_frame[f_idx] = []
                    by_frame[f_idx].append({
                        "track_id": item_idx,
                        "type": kitti_type,
                        "position_cam2": pos_cam2,
                        "X": float(pos_cam2[0]),
                        "Y": float(pos_cam2[1]),
                        "Z": float(pos_cam2[2]),
                        "distance_m": float(np.linalg.norm(pos_cam2))
                    })

        return by_frame


def run_kitti_evaluation(num_frames: int = 50, output_json: Optional[str] = None):
    print("\n=======================================================")
    print(" BUSSENSE EVALUATION — KITTI STEREO TRAFFIC BENCHMARK")
    print("=======================================================")

    kitti_base = Path(r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\data\kitti")
    calib_dir = kitti_base / "2011_09_26_calib"
    seq_dir = kitti_base / "2011_09_26_drive_0005"
    tracklets_file = seq_dir / "tracklet_labels.xml"

    left_files = sorted(list((seq_dir / "image_02" / "data").glob("*.png")))
    right_files = sorted(list((seq_dir / "image_03" / "data").glob("*.png")))

    gt_evaluator = KittiTrackletEvaluator(calib_dir, tracklets_file)

    fx = gt_evaluator.fx
    fy = gt_evaluator.fy
    cx = gt_evaluator.cx
    cy = gt_evaluator.cy
    baseline = gt_evaluator.baseline

    print(f"Loaded Calibration: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, B={baseline:.4f}m")
    print(f"Total Available Frames: {len(left_files)}")

    eval_frames = min(num_frames, len(left_files))

    # Initialize modules
    matcher = get_stereo_matcher("sgbm")  # Fast and reliable for benchmark
    tracker = TrafficTracker()
    analyzer = TrafficSpatialAnalyzer(
        fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline, min_valid_disparity=0.5
    )
    density_eng = TrafficDensityEngine(
        fx=fx, img_width=1242, img_height=375, z_min=2.0, z_max=35.0
    )

    total_time = 0.0
    total_preds = 0
    total_gt = 0
    true_positives = 0
    false_positives = 0
    distance_errors = []
    depth_z_errors = []
    id_switches = 0
    prev_track_id_map = {}  # gt_id -> pred_id

    calib_dict = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline}

    print(f"Evaluating {eval_frames} synchronized stereo frames...\n")

    for i in range(eval_frames):
        left_img = cv2.imread(str(left_files[i]))
        right_img = cv2.imread(str(right_files[i]))

        t0 = time.time()
        # 1. DISPARITY (ONLY LEFT + RIGHT RGB)
        disp = matcher.estimate_disparity(left_img, right_img)
        # 2. TRACKING
        tracks = tracker.track(left_img, frame_index=i)
        # 3. 3D POSITIONING
        scene = analyzer.analyze_scene(tracks, disp)
        t1 = time.time()

        total_time += (t1 - t0)

        # 4. EVALUATION: Compare predictions with GT
        gt_objs = gt_evaluator.tracklets_by_frame.get(i, [])
        # Focus on vehicle GT classes
        gt_vehicles = [g for g in gt_objs if g["type"] in ["Car", "Van", "Truck"]]
        total_gt += len(gt_vehicles)

        pred_vehicles = [v for v in scene["vehicles"] if v["valid"]]
        total_preds += len(pred_vehicles)

        matched_gt = set()
        for pred in pred_vehicles:
            pred_type = pred["class"]
            pred_dist = pred["distance_m"]
            pred_z = pred["Z_m"]
            pred_id = pred.get("track_id", -1)

            target_kitti_types = YOLO_TO_KITTI_MAP.get(pred_type, ["Car", "Van"])

            best_err = None
            best_gt_idx = None
            for g_idx, g in enumerate(gt_vehicles):
                if g_idx in matched_gt:
                    continue
                if g["type"] in target_kitti_types:
                    err = abs(pred_dist - g["distance_m"])
                    if best_err is None or err < best_err:
                        best_err = err
                        best_gt_idx = g_idx

            if best_err is not None and best_err < 4.0:  # 4m association threshold
                true_positives += 1
                matched_gt.add(best_gt_idx)
                distance_errors.append(best_err)
                gt_match = gt_vehicles[best_gt_idx]
                depth_z_errors.append(abs(pred_z - gt_match["Z"]))

                gt_track = gt_match["track_id"]
                if gt_track in prev_track_id_map and prev_track_id_map[gt_track] != pred_id:
                    id_switches += 1
                prev_track_id_map[gt_track] = pred_id
            else:
                false_positives += 1

    precision = true_positives / max(1, total_preds)
    recall = true_positives / max(1, total_gt)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    mean_dist_err = float(np.mean(distance_errors)) if distance_errors else 0.0
    rmse_dist_err = float(np.sqrt(np.mean(np.array(distance_errors)**2))) if distance_errors else 0.0
    mean_z_err = float(np.mean(depth_z_errors)) if depth_z_errors else 0.0
    fps = eval_frames / max(1e-3, total_time)

    results = {
        "dataset": "KITTI Tracking",
        "sequence": "2011_09_26_drive_0005",
        "frames_evaluated": eval_frames,
        "runtime_s": round(total_time, 2),
        "fps": round(fps, 1),
        "detection_metrics": {
            "total_predictions": total_preds,
            "total_ground_truth": total_gt,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3)
        },
        "tracking_metrics": {
            "id_switches": id_switches,
            "tracking_continuity": round(1.0 - (id_switches / max(1, true_positives)), 3)
        },
        "stereo_3d_metrics": {
            "num_distance_evaluations": len(distance_errors),
            "distance_mae_m": round(mean_dist_err, 2),
            "distance_rmse_m": round(rmse_dist_err, 2),
            "longitudinal_depth_z_mae_m": round(mean_z_err, 2)
        }
    }

    print("\n[KITTI Traffic Evaluation Results]")
    print(f"  Frames Evaluated:        {eval_frames}")
    print(f"  Inference Speed:         {fps:.1f} FPS")
    print(f"  Precision:               {precision:.1%}")
    print(f"  Recall:                  {recall:.1%}")
    print(f"  F1 Score:                {f1:.3f}")
    print(f"  ID Switches:             {id_switches}")
    print(f"  Tracking Continuity:     {results['tracking_metrics']['tracking_continuity']:.1%}")
    print(f"  3D Distance MAE:         {mean_dist_err:.2f} m")
    print(f"  3D Distance RMSE:        {rmse_dist_err:.2f} m")
    print(f"  Longitudinal Z MAE:      {mean_z_err:.2f} m")
    print("=======================================================\n")

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_json}")

    return results


if __name__ == "__main__":
    out_file = PROJECT_ROOT / "outputs" / "evaluation" / "kitti_traffic_eval.json"
    run_kitti_evaluation(num_frames=50, output_json=str(out_file))
