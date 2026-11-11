"""
Final Autonomous Validation Video Demo Renderer — Phase 11
Renders a 40–60 second multi-sequence full-HUD demonstration video per PRD §68.
Shows actual stereo 3D localization, persistent tracking, safety corridor projection,
VRU in-path/crossing alerts, and lead vehicle kinematics without fabricated GT overlays.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.calibration import KittiCalibration
from scripts.stereo_sgbm import SGBMStereoEngine
from scripts.detector import ObjectDetector
from scripts.vehicle_tracker import SceneTracker
from scripts.ego_motion import EgoMotionProvider
from scripts.kinematics_engine import KinematicsEngine
from scripts.vru_engine import VRUEngine
from scripts.incident_engine import IncidentEngine


def draw_projected_corridor(image: np.ndarray, calib: KittiCalibration, corridor_half_w: float = 1.8, max_z: float = 25.0):
    """Draws ground plane projected bus corridor on the image."""
    z_steps = np.linspace(2.0, max_z, 20)
    pts_left = []
    pts_right = []

    for z in z_steps:
        # Ground plane in camera coordinates (Y ~ 1.65m below camera optical center)
        y_ground = 1.65
        # Left boundary: X = -corridor_half_w
        u_l, v_l, _ = calib.project_rect_cam_to_image2(np.array([[-corridor_half_w, y_ground, z]]))
        # Right boundary: X = +corridor_half_w
        u_r, v_r, _ = calib.project_rect_cam_to_image2(np.array([[corridor_half_w, y_ground, z]]))

        if 0 <= u_l[0] < calib.image_width and 0 <= v_l[0] < calib.image_height:
            pts_left.append([int(u_l[0]), int(v_l[0])])
        if 0 <= u_r[0] < calib.image_width and 0 <= v_r[0] < calib.image_height:
            pts_right.append([int(u_r[0]), int(v_r[0])])

    if len(pts_left) > 2 and len(pts_right) > 2:
        poly = np.array(pts_left + pts_right[::-1], dtype=np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [poly], (0, 180, 0))  # Translucent green corridor
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
        cv2.polylines(image, [np.array(pts_left, dtype=np.int32)], False, (0, 255, 0), 2)
        cv2.polylines(image, [np.array(pts_right, dtype=np.int32)], False, (0, 255, 0), 2)


def render_demo():
    print("============================================================")
    print("STARTING PHASE 11: RENDERING FINAL VALIDATION VIDEO DEMO")
    print("============================================================")

    base_dir = r"C:\Users\Integra\.gemini\antigravity\scratch\kitti_incident_intelligence"
    kitti_base = r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\data\kitti"
    calib_cam_path = os.path.join(kitti_base, "2011_09_26_calib", "calib_cam_to_cam.txt")
    calib_velo_path = os.path.join(kitti_base, "2011_09_26_calib", "calib_velo_to_cam.txt")
    weights_path = r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\yolov8n.pt"
    demo_out_dir = os.path.join(base_dir, "outputs", "demo")
    os.makedirs(demo_out_dir, exist_ok=True)
    out_video_path = os.path.join(demo_out_dir, "kitti_incident_vru_demo.mp4")

    # Load Calibration
    calib = KittiCalibration(calib_cam_path, calib_velo_path)
    stereo_engine = SGBMStereoEngine(minDisparity=0, numDisparities=128, blockSize=5)
    detector = ObjectDetector(weights_path=weights_path, conf_threshold=0.30)
    kinematics_engine = KinematicsEngine(max_kinematic_range=30.0)
    vru_engine = VRUEngine(calib=calib, corridor_max_range=25.0)
    incident_engine = IncidentEngine(max_validated_range_m=25.0)

    # Use multiple sequences for rich multi-interaction demo (PRD §68)
    sequences = [
        ("2011_09_26_drive_0002_sync", "KITTI SEQ 0001 (VRU, Cyclists, Pedestrians)"),
        ("2011_09_26_drive_0013_sync", "KITTI SEQ 0003 (Lead Vehicle Following, Urban Traffic)"),
        ("2011_09_26_drive_0017_sync", "KITTI SEQ 0005 (Changing Urban Traffic Dynamics)"),
    ]

    # Target 8 FPS output: ~335 frames total -> ~42 seconds
    target_fps = 8.0
    H, W = calib.image_height, calib.image_width
    hud_h = H + 80  # extra header bar
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_video_path, fourcc, target_fps, (W, hud_h))

    total_rendered_frames = 0

    for seq_folder, seq_desc in sequences:
        seq_path = os.path.join(kitti_base, "2011_09_26", seq_folder)
        left_dir = os.path.join(seq_path, "image_02", "data")
        right_dir = os.path.join(seq_path, "image_03", "data")
        oxts_dir = os.path.join(seq_path, "oxts")

        if not os.path.exists(left_dir) or not os.path.exists(right_dir):
            continue

        tracker = SceneTracker(calib=calib, dt=0.1)
        ego_provider = EgoMotionProvider(oxts_dir=oxts_dir, nominal_dt=0.1) if os.path.exists(oxts_dir) else None

        frame_files = sorted(os.listdir(left_dir))
        print(f"[DEMO] Rendering {seq_folder} ({len(frame_files)} frames)...")

        for frame_idx, fname in enumerate(frame_files):
            timestamp = frame_idx * 0.1
            left_img = cv2.imread(os.path.join(left_dir, fname))
            right_img = cv2.imread(os.path.join(right_dir, fname))

            # Disparity pass
            disp, _ = stereo_engine.compute_disparity(left_img, right_img)
            depth_map = stereo_engine.disparity_to_depth(disp, calib.fB)

            # Detection & Tracking
            detections = detector.detect(left_img)
            active_tracks = tracker.update(detections, depth_map, frame_idx, timestamp)

            # Telemetry
            ego_frame = ego_provider.get_frame(frame_idx) if ego_provider else None
            bus_speed_kmh = (ego_frame.speed_mps * 3.6) if ego_frame else 0.0
            bus_speed_mps = ego_frame.speed_mps if ego_frame else 0.0

            # Kinematics
            lead_trk = kinematics_engine.identify_lead_vehicle(active_tracks)
            lead_metrics = kinematics_engine.compute_lead_vehicle_metrics(lead_trk, ego_speed_mps=bus_speed_mps)

            # VRU Intelligence
            vru_obs_list = vru_engine.evaluate_vru_tracks(active_tracks, ego_speed_mps=bus_speed_mps)

            # Traffic Events
            traffic_events = incident_engine.evaluate_traffic_events(
                tracks=active_tracks,
                lead_metrics=lead_metrics,
                ego_frame=ego_frame,
                frame_id=frame_idx,
                timestamp=timestamp,
            )

            # -------------------------------------------------------------
            # Canvas Composition
            # -------------------------------------------------------------
            frame_canvas = np.zeros((hud_h, W, 3), dtype=np.uint8)
            canvas_scene = left_img.copy()

            # 1. Draw Safety Travel Corridor
            draw_projected_corridor(canvas_scene, calib, corridor_half_w=1.8, max_z=22.0)

            # 2. Draw Tracked Bounding Boxes & 3D Telemetry
            for trk in active_tracks:
                x1, y1, x2, y2 = trk.bbox.astype(int)
                x_3d, z_3d = trk.current_position_3d
                vx, vz = trk.current_velocity_3d

                if trk.category == "vru":
                    box_color = (0, 255, 255)  # Yellow for VRUs
                else:
                    box_color = (0, 220, 0) if trk != lead_trk else (0, 140, 255)  # Orange for lead car

                cv2.rectangle(canvas_scene, (x1, y1), (x2, y2), box_color, 2)

                label = f"#{trk.track_id} {trk.cls_name.upper()} | {z_3d:.1f}m"
                cv2.rectangle(canvas_scene, (x1, max(0, y1 - 20)), (x1 + len(label) * 9, y1), (20, 20, 20), -1)
                cv2.putText(canvas_scene, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # 3. Draw Active Event Banners on Scene
            banner_y = 40
            # A. VRU In Path / Crossing Banners
            for v_obs in vru_obs_list:
                for ev in v_obs["events"]:
                    etype = ev["event_type"]
                    alert_text = f"ALERT: {etype} | CLASS: {v_obs['vru_class'].upper()} | DIST: {v_obs['distance_m']:.1f}m | STATUS: UNVERIFIED"
                    cv2.rectangle(canvas_scene, (10, banner_y - 25), (W - 10, banner_y + 8), (0, 0, 180), -1)
                    cv2.putText(canvas_scene, alert_text, (20, banner_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    banner_y += 38

            # B. Traffic Incident Banners
            for te in traffic_events:
                tetype = te["event_type"]
                trk_id = te["vehicles"][0]["track_id"]
                dist = te["metrics"]["distance_m"]
                alert_text = f"INCIDENT: {tetype} | TRACK #{trk_id} | DIST: {dist:.1f}m | STATUS: UNVERIFIED"
                cv2.rectangle(canvas_scene, (10, banner_y - 25), (W - 10, banner_y + 8), (0, 69, 255), -1)
                cv2.putText(canvas_scene, alert_text, (20, banner_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                banner_y += 38

            # Paste scene into canvas
            frame_canvas[80:, :] = canvas_scene

            # 4. Top Telemetry Header Bar
            cv2.rectangle(frame_canvas, (0, 0), (W, 80), (18, 18, 18), -1)
            cv2.line(frame_canvas, (0, 79), (W, 79), (50, 50, 50), 1)

            # System title
            cv2.putText(frame_canvas, "BUSSENSE ROAD SAFETY SENSING UNIT", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.putText(frame_canvas, f"AUTONOMOUS STANDALONE VALIDATION | {seq_desc}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Bus Telemetry Card
            lat_str = f"{ego_frame.lat:.5f} N" if ego_frame else "N/A"
            lon_str = f"{ego_frame.lon:.5f} E" if ego_frame else "N/A"
            cv2.putText(frame_canvas, f"BUS SPEED: {bus_speed_kmh:.1f} km/h", (620, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_canvas, f"GPS: {lat_str}, {lon_str}", (620, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Lead Vehicle Card
            if lead_metrics["lead_track_id"] is not None:
                gap_val = lead_metrics['gap_m']
                ttc_val = f"{lead_metrics['ttc_consensus_s']:.1f}s" if lead_metrics['ttc_consensus_s'] else "N/A"
                hw_val = f"{lead_metrics['time_headway_s']:.1f}s" if lead_metrics['time_headway_s'] else "N/A"
                cv2.putText(frame_canvas, f"LEAD GAP: {gap_val:.1f}m | TTC: {ttc_val}", (940, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
                cv2.putText(frame_canvas, f"HEADWAY: {hw_val} | CLOSING: {lead_metrics['closing_rate_mps']:.1f} m/s", (940, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            else:
                cv2.putText(frame_canvas, "LEAD VEHICLE: NONE IN PATH", (940, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

            # Write frame to video
            out_writer.write(frame_canvas)
            total_rendered_frames += 1

    out_writer.release()
    total_duration_s = total_rendered_frames / target_fps
    print(f"[DEMO] Video successfully saved to: {out_video_path}")
    print(f"[DEMO] Total Rendered Frames: {total_rendered_frames} | Duration: {total_duration_s:.1f} seconds at {target_fps} FPS.")


if __name__ == "__main__":
    render_demo()
