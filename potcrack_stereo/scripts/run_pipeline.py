"""
run_pipeline.py - BusSense Vehicle Stereo Pothole 3D Measurement End-to-End Pipeline

STRICT DEPLOYMENT PIPELINE:
  LEFT RGB + RIGHT RGB
          ↓
  stereo disparity/depth (Pluggable: Fast-FoundationStereo / SGBM)
          ↓
  LEFT RGB + predicted depth
          ↓
  pothole segmentation (Trained YOLO11-seg)
          ↓
  3D pothole ROI
          ↓
  local road-surface estimation (RANSAC plane fit)
          ↓
  pothole depth + footprint area (Calibrated ray intersection + Shoelace)
          ↓
  temporal tracking & measurement smoothing
          ↓
  pipeline JSON output & performance profiling
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.stereo_depth import FastFoundationStereoMatcher, SGBMStereoMatcher, disparity_to_depth
from scripts.pothole_geometry import PotholeGeometryEngine, PotholeInstance3D

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PotholeTracker:
    """
    Lightweight IoU and Centroid Tracker for consistent pothole tracking
    and measurement temporal smoothing.
    """

    def __init__(self, iou_thresh: float = 0.25, max_missing: int = 5, ema_alpha: float = 0.6):
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing
        self.ema_alpha = ema_alpha
        self.next_id = 1
        self.active_tracks: Dict[int, dict] = {}

    def _compute_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea

        return interArea / float(unionArea) if unionArea > 0 else 0.0

    def update(self, detected_instances: List[PotholeInstance3D]) -> List[dict]:
        """
        Matches detections to active tracks and smooths metrics.
        Returns list of track dictionaries.
        """
        matched_tracks = []
        unmatched_detections = list(range(len(detected_instances)))
        unmatched_tracks = set(self.active_tracks.keys())

        # Match by IoU
        for det_idx in list(unmatched_detections):
            det = detected_instances[det_idx]
            best_iou = 0.0
            best_trk_id = None

            for trk_id in unmatched_tracks:
                trk = self.active_tracks[trk_id]
                iou = self._compute_iou(det.bbox_xyxy, trk["bbox"])
                if iou > best_iou and iou >= self.iou_thresh:
                    best_iou = iou
                    best_trk_id = trk_id

            if best_trk_id is not None:
                # Update track with smoothed measurements
                trk = self.active_tracks[best_trk_id]
                alpha = self.ema_alpha
                
                # Smooth depth
                if det.depth_p95_cm is not None:
                    if trk["depth_p95_cm"] is not None:
                        smoothed_depth = alpha * det.depth_p95_cm + (1.0 - alpha) * trk["depth_p95_cm"]
                    else:
                        smoothed_depth = det.depth_p95_cm
                else:
                    smoothed_depth = trk["depth_p95_cm"]

                # Smooth area
                if det.area_m2 is not None:
                    if trk["area_m2"] is not None:
                        smoothed_area = alpha * det.area_m2 + (1.0 - alpha) * trk["area_m2"]
                    else:
                        smoothed_area = det.area_m2
                else:
                    smoothed_area = trk["area_m2"]

                trk.update({
                    "bbox": det.bbox_xyxy,
                    "contour_px": det.contour_px,
                    "mask": det.mask,
                    "confidence": det.confidence,
                    "depth_p95_cm": round(smoothed_depth, 1) if smoothed_depth is not None else None,
                    "area_m2": round(smoothed_area, 4) if smoothed_area is not None else None,
                    "length_m": det.length_m,
                    "width_m": det.width_m,
                    "quality": det.quality,
                    "quality_reason": det.quality_reason,
                    "missing_frames": 0,
                    "hits": trk["hits"] + 1,
                    "history_depth": trk["history_depth"] + [det.depth_p95_cm],
                    "history_area": trk["history_area"] + [det.area_m2],
                    "has_current_detection": True,
                    "is_renderable": (det.quality != "INVALID" and smoothed_depth is not None and smoothed_depth > 0 and smoothed_area is not None and smoothed_area > 0),
                })
                matched_tracks.append(trk)
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(best_trk_id)

        # Create new tracks for remaining detections
        for det_idx in unmatched_detections:
            det = detected_instances[det_idx]
            is_rend = (det.quality != "INVALID" and det.depth_p95_cm is not None and det.depth_p95_cm > 0 and det.area_m2 is not None and det.area_m2 > 0)
            new_trk = {
                "track_id": self.next_id,
                "bbox": det.bbox_xyxy,
                "contour_px": det.contour_px,
                "mask": det.mask,
                "confidence": det.confidence,
                "depth_p95_cm": det.depth_p95_cm,
                "area_m2": det.area_m2,
                "length_m": det.length_m,
                "width_m": det.width_m,
                "quality": det.quality,
                "quality_reason": det.quality_reason,
                "missing_frames": 0,
                "hits": 1,
                "history_depth": [det.depth_p95_cm] if det.depth_p95_cm is not None else [],
                "history_area": [det.area_m2] if det.area_m2 is not None else [],
                "has_current_detection": True,
                "is_renderable": is_rend,
            }
            self.active_tracks[self.next_id] = new_trk
            matched_tracks.append(new_trk)
            self.next_id += 1

        # Increment missing frames for unobserved tracks
        for trk_id in unmatched_tracks:
            self.active_tracks[trk_id]["missing_frames"] += 1
            self.active_tracks[trk_id]["has_current_detection"] = False
            self.active_tracks[trk_id]["is_renderable"] = False

        # Remove dead tracks
        dead_tracks = [t_id for t_id, t in self.active_tracks.items() if t["missing_frames"] > self.max_missing]
        for t_id in dead_tracks:
            del self.active_tracks[t_id]

        return matched_tracks


class BusSenseStereoPipeline:
    """
    End-to-End Stereo Pothole 3D Metrology Pipeline.
    """

    def __init__(
        self,
        stereo_backend: str = "fast_foundation",
        seg_weights: Optional[Path] = None,
        seg_conf: float = 0.20,
    ):
        self.stereo_backend_name = stereo_backend
        if stereo_backend == "fast_foundation":
            self.stereo_matcher = FastFoundationStereoMatcher()
            if self.stereo_matcher.model is None:
                print("[Pipeline] Falling back to SGBM matcher")
                self.stereo_matcher = SGBMStereoMatcher()
        else:
            self.stereo_matcher = SGBMStereoMatcher()

        # Segmentation model
        default_seg_weights = PROJECT_ROOT / "checkpoints" / "segmentation" / "best_pothole_seg.pt"
        self.seg_weights_path = seg_weights or default_seg_weights
        if self.seg_weights_path.exists():
            print(f"[Pipeline] Loading YOLO11-seg model from {self.seg_weights_path}")
            self.seg_model = YOLO(str(self.seg_weights_path))
        else:
            print(f"[Pipeline] Warning: Custom seg weights not found at {self.seg_weights_path}, using base yolo11n-seg")
            self.seg_model = YOLO("yolo11n-seg.pt")

        self.seg_conf = seg_conf
        self.geom_engine = PotholeGeometryEngine()
        self.tracker = PotholeTracker()

    def process_frame(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        calib: dict,
    ) -> dict:
        """
        Executes single frame inference:
        LEFT + RIGHT -> Stereo Depth -> Pothole Seg -> 3D Metrology -> Tracking.
        """
        timings = {}

        # 1. Stereo Disparity & Depth
        t0 = time.time()
        pred_disp = self.stereo_matcher.estimate_disparity(left_img, right_img)
        pred_depth, _ = disparity_to_depth(pred_disp, calib)
        timings["stereo_ms"] = (time.time() - t0) * 1000.0

        # 2. Pothole Segmentation
        t1 = time.time()
        h, w = left_img.shape[:2]
        seg_results = self.seg_model.predict(
            left_img, conf=self.seg_conf, imgsz=640, device=0, verbose=False
        )[0]

        detected_items = []
        if seg_results.masks is not None and len(seg_results.masks) > 0:
            masks_data = seg_results.masks.data.cpu().numpy() # (N, H_mask, W_mask)
            boxes_data = seg_results.boxes.xyxy.cpu().numpy()
            confs_data = seg_results.boxes.conf.cpu().numpy()

            for i in range(len(masks_data)):
                m_i = masks_data[i]
                if m_i.shape != (h, w):
                    m_i = cv2.resize((m_i > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    m_i = (m_i > 0.5).astype(np.uint8)

                box_i = tuple(map(int, boxes_data[i]))
                conf_i = float(confs_data[i])

                detected_items.append({
                    "mask": m_i,
                    "bbox": box_i,
                    "confidence": conf_i
                })
        timings["seg_ms"] = (time.time() - t1) * 1000.0

        # 3. 3D Pothole Geometry Engine
        t2 = time.time()
        pothole_3d_instances = self.geom_engine.analyze_potholes(
            left_img, pred_depth, detected_items, calib
        )
        timings["geometry_ms"] = (time.time() - t2) * 1000.0

        # 4. Temporal Tracking & Smoothing
        t3 = time.time()
        tracked_potholes = self.tracker.update(pothole_3d_instances)
        timings["tracking_ms"] = (time.time() - t3) * 1000.0

        timings["total_ms"] = sum(timings.values())
        timings["fps"] = 1000.0 / timings["total_ms"] if timings["total_ms"] > 0 else 0.0

        return {
            "pred_disparity": pred_disp,
            "pred_depth": pred_depth,
            "tracked_potholes": tracked_potholes,
            "raw_instances": pothole_3d_instances,
            "timings": timings
        }


def run_pipeline_on_sequence(
    sequence_dir: Path,
    calib_path: Path,
    output_dir: Path = OUTPUT_DIR,
    max_frames: int = 50,
    stereo_backend: str = "fast_foundation",
) -> dict:
    """
    Runs the pipeline on a stereo sequence, recording full metrology and timings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(calib_path, "rb") as f:
        calib = pickle.load(f)

    pipeline = BusSenseStereoPipeline(stereo_backend=stereo_backend)

    # Discover stereo pairs
    left_files = sorted(list((sequence_dir / "left_half").glob("*.jpg")))
    if not left_files:
        left_files = sorted(list(sequence_dir.glob("*_i.jpg")))
        is_rsrd = False
    else:
        is_rsrd = True

    selected_files = left_files[:max_frames]
    print(f"[Pipeline] Processing {len(selected_files)} stereo frames using backend: {stereo_backend}")

    frame_results = []
    all_timings = []

    for idx, lf in enumerate(selected_files):
        stem = lf.stem
        if is_rsrd:
            rf = sequence_dir / "right_half" / f"{stem}.jpg"
        else:
            rf = lf.parent / f"{stem[:-2]}_d.jpg"

        left_img = cv2.imread(str(lf))
        right_img = cv2.imread(str(rf))
        if left_img is None or right_img is None:
            continue

        res = pipeline.process_frame(left_img, right_img, calib)
        all_timings.append(res["timings"])

        # Serializable records
        frame_record = {
            "frame_idx": idx,
            "frame_stem": stem,
            "timings": res["timings"],
            "potholes": [
                {
                    "track_id": p["track_id"],
                    "bbox": list(p["bbox"]),
                    "confidence": round(p["confidence"], 3),
                    "depth_p95_cm": p["depth_p95_cm"],
                    "area_m2": p["area_m2"],
                    "length_m": p["length_m"],
                    "width_m": p["width_m"],
                    "quality": p["quality"],
                    "quality_reason": p["quality_reason"]
                }
                for p in res["tracked_potholes"]
            ]
        }
        frame_results.append(frame_record)

        if (idx + 1) % 5 == 0 or idx == len(selected_files) - 1:
            print(f"  Frame [{idx+1}/{len(selected_files)}] {stem}: "
                  f"Potholes={len(res['tracked_potholes'])} | "
                  f"Total Latency={res['timings']['total_ms']:.1f} ms ({res['timings']['fps']:.1f} FPS)")

    # Latency Summary
    latency_summary = {
        "mean_stereo_ms": float(np.mean([t["stereo_ms"] for t in all_timings])),
        "mean_seg_ms": float(np.mean([t["seg_ms"] for t in all_timings])),
        "mean_geometry_ms": float(np.mean([t["geometry_ms"] for t in all_timings])),
        "mean_tracking_ms": float(np.mean([t["tracking_ms"] for t in all_timings])),
        "mean_total_ms": float(np.mean([t["total_ms"] for t in all_timings])),
        "mean_fps": float(np.mean([t["fps"] for t in all_timings])),
    }

    report = {
        "stereo_backend": stereo_backend,
        "total_frames_processed": len(frame_results),
        "latency_profile": latency_summary,
        "frame_records": frame_results
    }

    with open(output_dir / "pipeline_measurements.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n------------------------------------------------------------")
    print("PIPELINE EXECUTION COMPLETE:")
    print("------------------------------------------------------------")
    print(f"Stereo Latency   : {latency_summary['mean_stereo_ms']:.1f} ms")
    print(f"Seg Latency      : {latency_summary['mean_seg_ms']:.1f} ms")
    print(f"Geometry Latency : {latency_summary['mean_geometry_ms']:.1f} ms")
    print(f"Total Latency    : {latency_summary['mean_total_ms']:.1f} ms ({latency_summary['mean_fps']:.1f} FPS)")
    print(f"Output Saved to  : {output_dir / 'pipeline_measurements.json'}")
    print("============================================================\n")

    return report


if __name__ == "__main__":
    rsrd_dir = PROJECT_ROOT / "data" / "RSRD"
    calib = rsrd_dir / "calibration_files" / "calib_20230317.pkl"
    run_pipeline_on_sequence(rsrd_dir, calib, max_frames=30)
