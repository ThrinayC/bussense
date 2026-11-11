"""
Monocular Hit-and-Run Investigation & ANPR Intelligence Engine
Processes monocular collision, hit-and-run, and critical traffic footage.
Explicit modality: source_modality = "monocular", metric_depth_valid = false.

Detects:
1. Contact / collision candidates between vehicle and victim (pedestrian/cyclist or vehicle)
2. Victim state transition (sudden aspect ratio collapse / fallen pose / abrupt standstill)
3. Suspect vehicle non-stop departure / rapid scene exit
4. Forensic ANPR license plate extraction and enhancement on suspect vehicle
5. Comprehensive evidence packaging (event.json, ocr.json, crops, video snippets)
"""

import os
import sys
import re
import json
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.detector import ObjectDetector, Detection
from scripts.anpr_engine import ANPREngine
from scripts.robust_2d_tracker import Robust2DTracker, TrackQualityState, Tracked2DObject


def compute_2d_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return float(interArea / (boxAArea + boxBArea - interArea))


class MonocularTrack:
    """Persistent 2D track for monocular video analysis."""

    def __init__(self, track_id: int, det: Detection, frame_idx: int):
        self.track_id = track_id
        self.cls_name = det.cls_name
        self.category = det.category
        self.bbox = det.bbox.copy()
        self.conf = det.conf
        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1
        self.time_since_update = 0

        # Motion & state history
        self.history: List[Dict] = []
        self._record(frame_idx)

        # ANPR fields
        self.plate_text: Optional[str] = None
        self.plate_confidence: float = 0.0
        self.plate_verified: bool = False
        self.plate_crops: List[np.ndarray] = []
        self.vehicle_crops: List[np.ndarray] = []

    def _record(self, frame_idx: int):
        x1, y1, x2, y2 = self.bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        aspect = h / w

        vx, vy, speed = 0.0, 0.0, 0.0
        if len(self.history) > 0:
            prev = self.history[-1]
            dt = max(1, frame_idx - prev["frame_idx"])
            vx = (cx - prev["cx"]) / dt
            vy = (cy - prev["cy"]) / dt
            speed = float(np.hypot(vx, vy))

        self.history.append({
            "frame_idx": frame_idx,
            "bbox": self.bbox.tolist(),
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aspect": aspect,
            "vx": vx,
            "vy": vy,
            "speed": speed,
        })
        if len(self.history) > 120:
            self.history.pop(0)

    def update(self, det: Detection, frame_idx: int, frame_img: Optional[np.ndarray] = None):
        self.bbox = det.bbox.copy()
        self.conf = det.conf
        self.cls_name = det.cls_name
        self.last_frame = frame_idx
        self.hits += 1
        self.time_since_update = 0
        self._record(frame_idx)

        # Store vehicle crop sample if large enough
        if frame_img is not None and self.category == "vehicle":
            x1, y1, x2, y2 = det.bbox.astype(int)
            h_img, w_img = frame_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
            if (x2 - x1) >= 60 and (y2 - y1) >= 40:
                crop = frame_img[y1:y2, x1:x2].copy()
                if len(self.vehicle_crops) < 15:
                    self.vehicle_crops.append(crop)
                else:
                    self.vehicle_crops[np.random.randint(0, 15)] = crop

    def update_plate_observation(self, text: str, conf: float, plate_crop: Optional[np.ndarray] = None):
        norm = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(norm) >= 4:
            if len(norm) == 7 and norm[:2].isalpha() and norm[2:].isdigit():
                self.plate_text = f"{norm[:2]} {norm[2:]}"
            elif len(norm) == 6 and norm[:2].isalpha() and norm[2:].isdigit():
                self.plate_text = f"{norm[:2]} {norm[2:]}"
            elif len(norm) == 7 and norm[:3].isalpha() and norm[3:].isdigit():
                self.plate_text = f"{norm[:3]} {norm[3:]}"
            elif len(norm) == 8 and norm[:2].isalpha() and norm[2:5].isdigit() and norm[5:].isalpha():
                self.plate_text = f"{norm[:2]} {norm[2:5]} {norm[5:]}"
            elif ' ' in text:
                self.plate_text = ' '.join(text.strip().split())
            else:
                self.plate_text = norm
            self.plate_confidence = max(self.plate_confidence, conf)
            self.plate_verified = True
            if plate_crop is not None:
                self.plate_crops.append(plate_crop)


class MonocularHitAndRunEngine:
    """
    Monocular Hit-and-Run Event Classifier and Evidence Packaging Engine.
    """

    def __init__(self, detector: ObjectDetector, anpr_engine: ANPREngine):
        self.detector = detector
        self.anpr_engine = anpr_engine
        self.tracker = Robust2DTracker(
            high_conf_thresh=0.35, low_conf_thresh=0.15, iou_match_thresh=0.25, max_lost_frames=25
        )
        self.confirmed_incidents: List[Dict] = []

    def _associate_and_update(
        self,
        detections: List[Detection],
        frame_idx: int,
        frame_img: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> List[Tracked2DObject]:
        ts = timestamp if timestamp is not None else (frame_idx * 0.033)
        h_img, w_img = frame_img.shape[:2]
        renderable_tracks = self.tracker.update(
            detections, timestamp=ts, frame_idx=frame_idx,
            source_shape=(h_img, w_img), render_shape=(h_img, w_img)
        )
        # Store vehicle crop sample if large enough
        for trk in self.tracker.get_all_tracks():
            if trk.category == "vehicle" and trk.time_since_update == 0:
                x1, y1, x2, y2 = trk.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
                if (x2 - x1) >= 60 and (y2 - y1) >= 40:
                    crop = frame_img[y1:y2, x1:x2].copy()
                    if len(trk.vehicle_crops) < 15:
                        trk.vehicle_crops.append(crop)
                    else:
                        trk.vehicle_crops[np.random.randint(0, 15)] = crop
        return renderable_tracks

    def evaluate_scene(
        self,
        frame_idx: int,
        active_tracks: List[MonocularTrack],
        frame_img: np.ndarray,
        source_name: str = "monocular",
    ) -> List[Dict]:
        """
        Evaluates potential collisions, victim knockdown / collapse, and suspect vehicle departures.
        """
        vehicles = [t for t in active_tracks if t.category == "vehicle" and t.hits >= 3]
        vrus = [t for t in active_tracks if t.category == "vru" and t.hits >= 2]
        all_participants = vehicles + vrus

        events_in_frame = []

        # Check vehicle-VRU interactions or vehicle-vehicle impacts
        for veh in vehicles:
            v_b = veh.bbox
            v_hist = veh.history
            if len(v_hist) < 3:
                continue
            curr_v_speed = v_hist[-1]["speed"]

            for target in all_participants:
                if target.track_id == veh.track_id:
                    continue

                t_b = target.bbox
                iou = compute_2d_iou(v_b, t_b)

                # Proximity check
                dx = abs(0.5 * (v_b[0] + v_b[2]) - 0.5 * (t_b[0] + t_b[2]))
                dy = abs(0.5 * (v_b[1] + v_b[3]) - 0.5 * (t_b[1] + t_b[3]))
                diag = np.hypot(dx, dy)
                min_span = max(v_b[2] - v_b[0], t_b[2] - t_b[0])

                is_contact = (iou > 0.04) or (diag < min_span * 0.75)

                if is_contact:
                    # Check victim response (aspect ratio collapse or sudden speed stop)
                    t_hist = target.history
                    victim_down = False
                    if len(t_hist) >= 3 and target.category == "vru":
                        aspect_init = t_hist[0]["aspect"]
                        aspect_now = t_hist[-1]["aspect"]
                        # Person aspect ratio collapses when fallen horizontally
                        if aspect_now < aspect_init * 0.70 or aspect_now < 0.90:
                            victim_down = True

                    # Check suspect departure: vehicle speed does not drop to zero
                    suspect_departing = curr_v_speed > 6.0

                    # Run ANPR attempt on suspect vehicle
                    self.anpr_engine.update_track_anpr(veh, frame_img, distance_m=None, timestamp=frame_idx * 0.033)

                    event_data = {
                        "event_type": "HIT_AND_RUN_CANDIDATE",
                        "frame_idx": frame_idx,
                        "timestamp_s": round(frame_idx * 0.033, 2),
                        "source_modality": "monocular",
                        "metric_depth_valid": False,
                        "suspect_vehicle_id": veh.track_id,
                        "victim_id": target.track_id,
                        "victim_type": target.cls_name,
                        "victim_fallen_status": victim_down,
                        "suspect_speed_px_frame": round(curr_v_speed, 1),
                        "suspect_departing": suspect_departing,
                        "license_plate": veh.plate_text,
                        "plate_confidence": round(veh.plate_confidence, 2) if veh.plate_confidence > 0 else None,
                        "plate_verified": veh.plate_verified,
                        "confidence_score": 0.88 if (victim_down and suspect_departing) else 0.72,
                        "source": source_name,
                    }
                    events_in_frame.append(event_data)
                    self.confirmed_incidents.append(event_data)

        return events_in_frame

    def export_evidence_package(
        self,
        event_dict: Dict,
        suspect_track: MonocularTrack,
        output_base_dir: str,
    ) -> str:
        """
        Exports standardized evidence package to outputs/evidence/hit_and_run/<event_id>/
        per PRD §36, §45.
        """
        ev_id = f"EV_{event_dict['source'].upper()}_F{event_dict['frame_idx']}"
        ev_dir = os.path.join(output_base_dir, ev_id)
        os.makedirs(ev_dir, exist_ok=True)

        # 1. Event JSON
        with open(os.path.join(ev_dir, "event.json"), "w") as f:
            json.dump(event_dict, f, indent=2)

        # 2. OCR JSON
        ocr_info = {
            "track_id": suspect_track.track_id,
            "registration_number": suspect_track.plate_text,
            "plate_confidence": suspect_track.plate_confidence,
            "plate_verified": suspect_track.plate_verified,
            "source_modality": "monocular",
            "metric_depth_valid": False,
            "status": "VERIFIED_READING" if suspect_track.plate_verified else "UNREADABLE_LOW_RESOLUTION",
        }
        with open(os.path.join(ev_dir, "ocr.json"), "w") as f:
            json.dump(ocr_info, f, indent=2)

        # 3. Crops
        if suspect_track.vehicle_crops:
            best_veh = suspect_track.vehicle_crops[-1]
            cv2.imwrite(os.path.join(ev_dir, "vehicle_crop.png"), best_veh)
            roi, enh = self.anpr_engine.extract_plate_roi(best_veh)
            cv2.imwrite(os.path.join(ev_dir, "plate_best_frame.png"), roi)
            cv2.imwrite(os.path.join(ev_dir, "plate_enhanced.png"), enh)

        return ev_dir


def process_hit_and_run_video(
    video_path: str,
    output_annotated_path: str,
    evidence_dir: str,
    max_frames: int = 400,
    stride: int = 1,
) -> Dict:
    """Processes a monocular hit-and-run video and generates annotated demonstration video."""
    weights_path = r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\yolov8n.pt"
    detector = ObjectDetector(weights_path=weights_path, conf_threshold=0.25)
    anpr = ANPREngine()
    engine = MonocularHitAndRunEngine(detector, anpr)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    target_fps = 15.0
    hud_h = H + 80
    os.makedirs(os.path.dirname(output_annotated_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_annotated_path, fourcc, target_fps, (W, hud_h))

    source_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_idx = 0
    rendered_frames = 0
    detected_events = []

    print(f"[HIT_AND_RUN] Processing {source_name} ({W}x{H})...")

    while cap.isOpened() and rendered_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        dets = detector.detect(frame)
        active_tracks = engine._associate_and_update(dets, frame_idx, frame)
        events = engine.evaluate_scene(frame_idx, active_tracks, frame, source_name=source_name)

        if events:
            detected_events.extend(events)

        # -----------------------------------------------------------------
        # Render Frame with Clean HUD
        # -----------------------------------------------------------------
        canvas = np.zeros((hud_h, W, 3), dtype=np.uint8)
        scene = frame.copy()

        # Render Bboxes
        for trk in active_tracks:
            x1, y1, x2, y2 = trk.bbox.astype(int)
            is_suspect = any(e.get("suspect_vehicle_id") == trk.track_id for e in events)
            is_victim = any(e.get("victim_id") == trk.track_id for e in events)

            if is_suspect:
                box_color = (0, 0, 255)  # Bright Red for Hit-and-run Suspect
                lbl_color = (0, 0, 255)
            elif is_victim:
                box_color = (0, 69, 255)  # Orange for victim
                lbl_color = (0, 69, 255)
            elif trk.category == "vru":
                box_color = (0, 255, 255)  # Yellow for VRUs
                lbl_color = (0, 255, 255)
            else:
                box_color = (0, 200, 0)
                lbl_color = (0, 200, 0)

            cv2.rectangle(scene, (x1, y1), (x2, y2), box_color, 2)

            # HUD Label: strict rule - plate ONLY if verified, else ID & class
            plate_lbl = ANPREngine.get_hud_plate_label(trk)
            if plate_lbl:
                label_text = f"#{trk.track_id} {trk.cls_name.upper()} | {plate_lbl}"
            else:
                label_text = f"#{trk.track_id} {trk.cls_name.upper()}"

            cv2.rectangle(scene, (x1, max(0, y1 - 22)), (x1 + len(label_text) * 10, y1), (15, 15, 15), -1)
            cv2.putText(scene, label_text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Active Event Banner
        if events:
            ev = events[0]
            banner = f"INCIDENT: HIT-AND-RUN CANDIDATE | SUSPECT #{ev['suspect_vehicle_id']} | VICTIM #{ev['victim_id']} ({ev['victim_type'].upper()})"
            cv2.rectangle(scene, (10, 15), (W - 10, 55), (0, 0, 180), -1)
            cv2.putText(scene, banner, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Bottom corner subtle source tag (NO title card)
        corner_tag = f"SOURCE: {source_name.upper()} | MODALITY: MONOCULAR CCTV/DASHCAM (METRIC DEPTH: N/A)"
        cv2.rectangle(scene, (10, H - 35), (len(corner_tag) * 11 + 15, H - 10), (10, 10, 10), -1)
        cv2.putText(scene, corner_tag, (15, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        # Place scene into canvas
        canvas[80:, :] = scene

        # Top Telemetry Header
        cv2.rectangle(canvas, (0, 0), (W, 80), (18, 18, 18), -1)
        cv2.line(canvas, (0, 79), (W, 79), (50, 50, 50), 1)
        cv2.putText(canvas, "BUSSENSE CRITICAL INCIDENT & HIT-AND-RUN INTELLIGENCE", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(canvas, f"MONOCULAR CRITICAL INCIDENT RECONSTRUCTION | {source_name}", (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Status badge
        status_txt = "INCIDENT ACTIVE: HIT & RUN" if events else "MONITORING SCENE TRAFFIC"
        status_col = (0, 0, 220) if events else (0, 180, 0)
        cv2.rectangle(canvas, (W - 320, 18), (W - 20, 62), status_col, -1)
        cv2.putText(canvas, status_txt, (W - 305, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        out_writer.write(canvas)
        rendered_frames += 1
        frame_idx += 1

    cap.release()
    out_writer.release()

    # Export evidence if events were caught
    evidence_package_dir = None
    if engine.confirmed_incidents:
        best_ev = engine.confirmed_incidents[0]
        suspect_id = best_ev["suspect_vehicle_id"]
        suspect_trk = next((t for t in engine.tracks if t.track_id == suspect_id), None)
        if suspect_trk:
            evidence_package_dir = engine.export_evidence_package(best_ev, suspect_trk, evidence_dir)

    print(f"[HIT_AND_RUN] Finished {source_name}: {rendered_frames} frames saved to {output_annotated_path}")
    return {
        "source": source_name,
        "video_output": output_annotated_path,
        "frames": rendered_frames,
        "events_count": len(engine.confirmed_incidents),
        "evidence_dir": evidence_package_dir,
    }
