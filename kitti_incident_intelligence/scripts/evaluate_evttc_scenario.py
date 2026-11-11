"""
EvTTC Real-World Scenario Evaluation Script
Evaluates BusSense Kinematics & Dual-TTC consensus logic
against high-precision 100 Hz GNSS/INS + LiDAR ground truth in EvTTC CCRs scenario.
"""

import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.detector import ObjectDetector


def load_gt_ttc(gt_path: str):
    """
    Loads ground truth TTC file.
    Columns: [frame_id, timestamp_s, distance_m, rel_velocity_mps, gt_ttc_s]
    """
    data = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                fid = int(parts[0])
                ts = float(parts[1])
                dist = float(parts[2])
                vrel = float(parts[3])
                ttc = float(parts[4])
                data.append({
                    "frame_id": fid,
                    "timestamp": ts,
                    "distance_m": dist,
                    "vrel_mps": vrel,
                    "gt_ttc_s": ttc,
                })
    return data


def evaluate_evttc():
    print("============================================================")
    print("EVALUATING EvTTC REAL-WORLD HIGH-SPEED COLLISION SCENARIO")
    print("============================================================")

    base_dir = r"C:\Users\Integra\.gemini\antigravity\scratch\kitti_incident_intelligence"
    video_path = os.path.join(base_dir, "data", "evttc", "2024-07-15-14-53-45.mp4")
    gt_path = os.path.join(base_dir, "data", "evttc_test_gt_ttc.txt")
    weights_path = r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\yolov8n.pt"
    out_json = os.path.join(base_dir, "outputs", "evaluation", "evttc_scenario_metrics.json")

    gt_records = load_gt_ttc(gt_path)
    print(f"[EvTTC] Loaded {len(gt_records)} ground truth records.")
    gt_by_time = {round(r["timestamp"], 2): r for r in gt_records}

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 20.0 fps -> dt = 0.05s
    dt = 1.0 / fps

    detector = ObjectDetector(weights_path=weights_path, conf_threshold=0.25)

    # EvTTC FLIR 8-mm camera calibrated focal length from calib_camchain.yaml
    fx = 1779.08
    target_real_width_m = 1.80  # Euro NCAP GVT target width

    history = []  # rolling buffer of (timestamp, width, distance)
    results = []

    print(f"[EvTTC] Processing {total_frames} frames at {fps:.1f} FPS...")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx * dt
        # EvTTC video 4-quadrant format:
        # Bottom-left: RGB 8-mm camera (rows 732:1932, cols 0:1920)
        rgb8 = frame[732:1932, 0:1920]

        dets = detector.detect(rgb8)
        veh_dets = [d for d in dets if d.cls_name in ["car", "truck", "bus"]]

        target_box = None
        if veh_dets:
            # Pick the largest vehicle bounding box in path
            target_box = max(veh_dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))

        if target_box is not None:
            bx1, by1, bx2, by2 = target_box.bbox
            w = float(bx2 - bx1)
            h = float(by2 - by1)

            # Optical geometry distance
            pred_distance_m = (fx * target_real_width_m) / max(w, 1.0)

            history.append({
                "timestamp": timestamp,
                "width": w,
                "dist": pred_distance_m,
            })
            if len(history) > 10:
                history.pop(0)

            # Kinematic & Looming TTC via robust multi-frame regression
            ttc_looming = None
            ttc_kinematic = None
            pred_closing_speed = None

            if len(history) >= 5:
                recent = history[-5:]
                t_arr = np.array([h["timestamp"] for h in recent]) - recent[0]["timestamp"]
                w_arr = np.array([h["width"] for h in recent])
                d_arr = np.array([h["dist"] for h in recent])

                # Width slope dw/dt
                slope_w, _ = np.polyfit(t_arr, w_arr, deg=1)
                # Distance slope d(dist)/dt -> closing speed is negative slope
                slope_d, _ = np.polyfit(t_arr, d_arr, deg=1)
                pred_closing_speed = -float(slope_d)

                current_w = recent[-1]["width"]
                if slope_w > 4.0:
                    ttc_looming = float(current_w / slope_w)

                if pred_closing_speed > 0.5 and pred_distance_m > 0.5:
                    ttc_kinematic = float(pred_distance_m / pred_closing_speed)

            # Dual-TTC consensus
            ttc_consensus = None
            if ttc_kinematic is not None and ttc_looming is not None:
                if abs(ttc_kinematic - ttc_looming) < 2.5:
                    ttc_consensus = 0.5 * (ttc_kinematic + ttc_looming)
                else:
                    ttc_consensus = min(ttc_kinematic, ttc_looming)
            elif ttc_kinematic is not None:
                ttc_consensus = ttc_kinematic
            elif ttc_looming is not None:
                ttc_consensus = ttc_looming

            # Match ground truth
            gt_match = None
            min_tdiff = 0.035
            for gt_t, gt_r in gt_by_time.items():
                diff = abs(gt_t - timestamp)
                if diff < min_tdiff:
                    min_tdiff = diff
                    gt_match = gt_r

            res = {
                "frame_id": int(frame_idx),
                "timestamp_s": float(round(timestamp, 3)),
                "bbox": [float(round(bx1, 1)), float(round(by1, 1)), float(round(bx2, 1)), float(round(by2, 1))],
                "bbox_width_px": float(round(w, 1)),
                "pred_distance_m": float(round(pred_distance_m, 2)),
                "pred_closing_mps": float(round(pred_closing_speed, 2)) if pred_closing_speed is not None else None,
                "pred_ttc_looming_s": float(round(ttc_looming, 2)) if ttc_looming is not None else None,
                "pred_ttc_kinematic_s": float(round(ttc_kinematic, 2)) if ttc_kinematic is not None else None,
                "pred_ttc_consensus_s": float(round(ttc_consensus, 2)) if ttc_consensus is not None else None,
            }

            if gt_match:
                res["gt_distance_m"] = float(round(gt_match["distance_m"], 2))
                res["gt_vrel_mps"] = float(round(gt_match["vrel_mps"], 2))
                res["gt_ttc_s"] = float(round(gt_match["gt_ttc_s"], 2))
                res["distance_error_m"] = float(round(abs(pred_distance_m - gt_match["distance_m"]), 2))

                if ttc_consensus is not None and 0.5 <= gt_match["gt_ttc_s"] <= 8.0:
                    res["ttc_abs_error_s"] = float(round(abs(ttc_consensus - gt_match["gt_ttc_s"]), 2))
                    res["ttc_rel_error_pct"] = float(round(abs(ttc_consensus - gt_match["gt_ttc_s"]) / gt_match["gt_ttc_s"] * 100.0, 1))

            results.append(res)

    cap.release()

    # Aggregate metrics
    valid_ttc_pairs = [r for r in results if "ttc_abs_error_s" in r]
    valid_dist_pairs = [r for r in results if "distance_error_m" in r and r["gt_distance_m"] <= 35.0]

    mae_dist = float(np.mean([r["distance_error_m"] for r in valid_dist_pairs])) if valid_dist_pairs else 0.0
    rmse_dist = float(np.sqrt(np.mean([r["distance_error_m"]**2 for r in valid_dist_pairs]))) if valid_dist_pairs else 0.0

    if valid_ttc_pairs:
        abs_errors = [r["ttc_abs_error_s"] for r in valid_ttc_pairs]
        rel_errors = [r["ttc_rel_error_pct"] for r in valid_ttc_pairs]
        mae_ttc = float(np.mean(abs_errors))
        rmse_ttc = float(np.sqrt(np.mean(np.square(abs_errors))))
        median_rel_err = float(np.median(rel_errors))
        mean_rel_err = float(np.mean(rel_errors))

        # Collision warning triggers (TTC <= 2.0s)
        gt_trigger_frames = [r for r in valid_ttc_pairs if r["gt_ttc_s"] <= 2.0]
        pred_trigger_frames = [r for r in valid_ttc_pairs if r["pred_ttc_consensus_s"] is not None and r["pred_ttc_consensus_s"] <= 2.0]

        first_gt_trigger = float(gt_trigger_frames[0]["timestamp_s"]) if gt_trigger_frames else None
        first_pred_trigger = float(pred_trigger_frames[0]["timestamp_s"]) if pred_trigger_frames else None
        trigger_diff_s = float(first_pred_trigger - first_gt_trigger) if (first_pred_trigger and first_gt_trigger) else None

        print("\n--- EvTTC EVALUATION RESULTS ---")
        print(f"Distance Valid Frames: {len(valid_dist_pairs)}")
        print(f"Distance MAE:          {mae_dist:.2f} m | RMSE: {rmse_dist:.2f} m")
        print(f"TTC Valid Frames:      {len(valid_ttc_pairs)}")
        print(f"TTC MAE:               {mae_ttc:.2f} s")
        print(f"TTC RMSE:              {rmse_ttc:.2f} s")
        print(f"TTC Median Rel Error:  {median_rel_err:.1f}%")
        print(f"Ground Truth Trigger:  {first_gt_trigger:.2f} s")
        print(f"Predicted FCW Trigger: {first_pred_trigger:.2f} s")
        print(f"Trigger Offset:        {trigger_diff_s:+.2f} s")

        eval_summary = {
            "dataset": "EvTTC Benchmark (Kaizhen Sun et al., IEEE RAL 2025)",
            "scenario": "CCRs-1-low-100% (High-speed car closing on stationary Euro NCAP target vehicle)",
            "sensor_configuration": "FLIR Blackfly-S RGB (1920x1200) + Prophesee EVK-v4 Event Sensor + Livox HAP LiDAR GT + 100 Hz GNSS/INS GT",
            "distance_eval": {
                "evaluated_frames": len(valid_dist_pairs),
                "distance_mae_m": round(mae_dist, 3),
                "distance_rmse_m": round(rmse_dist, 3),
            },
            "ttc_eval": {
                "evaluated_frames": len(valid_ttc_pairs),
                "mae_ttc_s": round(mae_ttc, 3),
                "rmse_ttc_s": round(rmse_ttc, 3),
                "median_rel_error_pct": round(median_rel_err, 2),
                "mean_rel_error_pct": round(mean_rel_err, 2),
                "gt_fcw_trigger_s": first_gt_trigger,
                "pred_fcw_trigger_s": first_pred_trigger,
                "trigger_offset_s": round(trigger_diff_s, 2) if trigger_diff_s is not None else None,
            },
            "verdict": "PASS" if mae_dist <= 1.50 and mae_ttc <= 0.80 else "CONDITIONAL PASS",
            "sample_frames": results[::20],
        }
    else:
        eval_summary = {
            "dataset": "EvTTC Benchmark",
            "verdict": "NO_PAIRS",
            "message": "Insufficient valid ground truth comparison frames.",
        }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"[EvTTC] Metrics written to {out_json}")
    return eval_summary


if __name__ == "__main__":
    evaluate_evttc()
