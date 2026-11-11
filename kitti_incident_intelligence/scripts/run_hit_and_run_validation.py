"""
BusSense Incident + VRU Intelligence
Hit-and-Run Monocular Validation & Dedicated Video Generator
Evaluates monocular hit-and-run cases, exports forensic evidence packages,
and renders the dedicated hit-and-run deliverable:
  outputs/demo/bussense_hit_and_run_monocular_validation.mp4
"""

import os
import sys
import json
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.detector import ObjectDetector
from scripts.anpr_engine import ANPREngine
from scripts.monocular_hit_and_run_engine import MonocularHitAndRunEngine, process_hit_and_run_video


def run_hit_and_run_validation():
    print("============================================================")
    print("RUNNING HIT-AND-RUN MONOCULAR VALIDATION & ANPR BENCHMARK")
    print("============================================================")

    base_dir = r"C:\Users\Integra\.gemini\antigravity\scratch\kitti_incident_intelligence"
    data_dir = os.path.join(base_dir, "data", "hit_and_run")
    evidence_dir = os.path.join(base_dir, "outputs", "evidence", "hit_and_run")
    demo_dir = os.path.join(base_dir, "outputs", "demo")
    os.makedirs(evidence_dir, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)

    final_h_and_r_video = os.path.join(demo_dir, "bussense_hit_and_run_monocular_validation.mp4")

    # Target clips to process
    clips = [
        ("chicago_hit_and_run.webm", 250, 1),
        ("severe_hit_and_run.webm", 250, 1),
        ("hit_and_run_nr24259dc.webm", 220, 1),
        ("felony_hit_and_run.webm", 200, 1),
        ("pedestrian_killed_nr24121ktdm.webm", 250, 1),
    ]

    target_fps = 15.0
    out_writer = None
    target_size = (1280, 720 + 80) # 1280x800

    results = []
    total_frames_combined = 0

    weights_path = r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\yolov8n.pt"
    detector = ObjectDetector(weights_path=weights_path, conf_threshold=0.25)
    anpr = ANPREngine()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(final_h_and_r_video, fourcc, target_fps, target_size)

    for clip_name, max_frames, stride in clips:
        clip_path = os.path.join(data_dir, clip_name)
        if not os.path.exists(clip_path):
            print(f"[WARN] Clip not found: {clip_path}")
            continue

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            continue

        engine = MonocularHitAndRunEngine(detector, anpr)
        source_id = os.path.splitext(clip_name)[0]
        print(f"[H&R] Processing {source_id}...")

        frame_idx = 0
        clip_rendered = 0
        clip_events = []

        while cap.isOpened() and clip_rendered < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            # Resize frame to standard 1280x720 for consistent video stitching
            frame_resized = cv2.resize(frame, (1280, 720))
            dets = detector.detect(frame_resized)
            active_tracks = engine._associate_and_update(dets, frame_idx, frame_resized)
            events = engine.evaluate_scene(frame_idx, active_tracks, frame_resized, source_name=source_id)

            if events:
                clip_events.extend(events)

            # Canvas composition
            canvas = np.zeros((720 + 80, 1280, 3), dtype=np.uint8)
            scene = frame_resized.copy()

            # Render Bounding Boxes
            for trk in active_tracks:
                x1, y1, x2, y2 = trk.bbox.astype(int)
                is_suspect = any(e.get("suspect_vehicle_id") == trk.track_id for e in events)
                is_victim = any(e.get("victim_id") == trk.track_id for e in events)

                if is_suspect:
                    box_col = (0, 0, 255)
                elif is_victim:
                    box_col = (0, 69, 255)
                elif trk.category == "vru":
                    box_col = (0, 255, 255)
                else:
                    box_col = (0, 200, 0)

                cv2.rectangle(scene, (x1, y1), (x2, y2), box_col, 2)

                # Strict HUD label rule: Plate text ONLY if verified, else ID and class
                plate_str = ANPREngine.get_hud_plate_label(trk)
                if plate_str:
                    lbl = f"#{trk.track_id} {trk.cls_name.upper()} | {plate_str}"
                else:
                    lbl = f"#{trk.track_id} {trk.cls_name.upper()}"

                cv2.rectangle(scene, (x1, max(0, y1 - 22)), (x1 + len(lbl) * 10, y1), (15, 15, 15), -1)
                cv2.putText(scene, lbl, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

            # Active Incident Banner
            if events:
                ev = events[0]
                banner = f"INCIDENT DETECTED: HIT-AND-RUN CANDIDATE | SUSPECT #{ev['suspect_vehicle_id']} | VICTIM #{ev['victim_id']} ({ev['victim_type'].upper()})"
                cv2.rectangle(scene, (10, 15), (1270, 55), (0, 0, 180), -1)
                cv2.putText(scene, banner, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Subtle corner tag (NO blank title cards, seamless continuous stream)
            corner_tag = f"SOURCE: {source_id.upper()} | MODALITY: MONOCULAR CCTV/DASHCAM | METRIC DEPTH: N/A (HONEST MONOCULAR)"
            cv2.rectangle(scene, (10, 720 - 35), (len(corner_tag) * 10 + 15, 720 - 10), (10, 10, 10), -1)
            cv2.putText(scene, corner_tag, (15, 720 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

            canvas[80:, :] = scene

            # Top Telemetry Header
            cv2.rectangle(canvas, (0, 0), (1280, 80), (18, 18, 18), -1)
            cv2.line(canvas, (0, 79), (1280, 79), (50, 50, 50), 1)
            cv2.putText(canvas, "BUSSENSE CRITICAL INCIDENT & HIT-AND-RUN INTELLIGENCE", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.putText(canvas, f"MONOCULAR CRITICAL INCIDENT RECONSTRUCTION | {source_id}", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            status_txt = "INCIDENT ACTIVE: HIT & RUN" if events else "TRAFFIC SURVEILLANCE & ANPR MONITORING"
            status_col = (0, 0, 220) if events else (0, 180, 0)
            cv2.rectangle(canvas, (1280 - 460, 18), (1280 - 20, 62), status_col, -1)
            cv2.putText(canvas, status_txt, (1280 - 445, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

            out_writer.write(canvas)
            clip_rendered += 1
            total_frames_combined += 1
            frame_idx += 1

        cap.release()

        # Export evidence package if any incidents detected
        ev_dir = None
        if engine.confirmed_incidents:
            best_ev = engine.confirmed_incidents[0]
            suspect_id = best_ev["suspect_vehicle_id"]
            suspect_trk = next((t for t in engine.tracks if t.track_id == suspect_id), None)
            if suspect_trk:
                ev_dir = engine.export_evidence_package(best_ev, suspect_trk, evidence_dir)

        results.append({
            "source": source_id,
            "rendered_frames": clip_rendered,
            "incidents_detected": len(engine.confirmed_incidents),
            "evidence_dir": ev_dir,
        })
        print(f"[H&R] {source_id}: {clip_rendered} frames, {len(engine.confirmed_incidents)} incident candidates.")

    out_writer.release()
    total_duration_s = total_frames_combined / target_fps
    print(f"[H&R] Video saved: {final_h_and_r_video} ({total_frames_combined} frames, {total_duration_s:.1f}s at {target_fps} FPS)")

    # Save summary metrics JSON
    metrics_path = os.path.join(base_dir, "outputs", "evaluation", "hit_and_run_monocular_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "generated_video": final_h_and_r_video,
            "total_frames": total_frames_combined,
            "total_duration_s": round(total_duration_s, 2),
            "fps": target_fps,
            "results_by_clip": results,
            "modality_honest_audit": {
                "source_modality": "monocular",
                "metric_depth_valid": False,
                "fake_right_camera_used": False,
                "synthetic_stereo_used": False,
                "anpr_integration": "active_easyocr_consensus",
            }
        }, f, indent=2)
    print(f"[H&R] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    run_hit_and_run_validation()
