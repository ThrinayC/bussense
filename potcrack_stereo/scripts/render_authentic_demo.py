import os
import sys
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO

def fit_plane_svd(pts: np.ndarray):
    """Fit 3D plane using SVD. Returns normal (a, b, c), d, centroid, and plane RMSE."""
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2, :]
    if normal[1] < 0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    dists = np.dot(pts, normal) + d
    rmse = float(np.sqrt(np.mean(dists**2)))
    return normal, d, centroid, rmse

def shoelace_and_dimensions_3d(pts_3d: np.ndarray, normal: np.ndarray):
    """
    Project 3D contour vertices onto fitted plane.
    Returns:
      area: float (m^2)
      length: float (m)
      width: float (m)
    """
    if len(pts_3d) < 3:
        return 0.0, 0.0, 0.0
    u_axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, u_axis)) > 0.9:
        u_axis = np.array([0.0, 1.0, 0.0])
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    u_axis = np.cross(v_axis, normal)
    u_axis /= np.linalg.norm(u_axis)
    
    pts_2d = np.stack([np.dot(pts_3d, u_axis), np.dot(pts_3d, v_axis)], axis=1)
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    pts_2d_cv = pts_2d.astype(np.float32)
    rect = cv2.minAreaRect(pts_2d_cv)
    w_box, h_box = rect[1]
    length = float(max(w_box, h_box))
    width = float(min(w_box, h_box))
    
    return float(area), length, width

def draw_rounded_rect(img, pt1, pt2, color, thickness=1, r=10):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)

def main():
    print("=== BUSSENSE AUTHENTIC STEREO DEMO GENERATION ===")
    
    base_dir = r"data\fan_road_surface\dataset_1"
    left_dir = os.path.join(base_dir, "left")
    right_dir = os.path.join(base_dir, "right")
    demo_out_dir = r"outputs\demo"
    os.makedirs(demo_out_dir, exist_ok=True)
    
    # Authentic Camera Intrinsics & Baseline (Fan Native Stereo Rig)
    fx, fy = 698.804085, 698.804085
    cx, cy = 634.69008, 339.49033
    baseline_m = 0.119588  # 119.588 mm
    fB_m_px = fx * baseline_m  # 83.56858 m*px
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=48,
        numDisparities=160,
        blockSize=7,
        P1=8 * 7**2,
        P2=32 * 7**2,
        disp12MaxDiff=2,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    pothole_model = YOLO("checkpoints/segmentation/best_pothole_seg.pt")
    crack_model = YOLO("checkpoints/segmentation/combined_pothole_crack/weights/best.pt")
    
    # Sorted continuous frames 1..35
    files = sorted([f for f in os.listdir(left_dir) if f.endswith(".png")], key=lambda x: int(os.path.splitext(x)[0]))
    total_frames = len(files)
    print(f"Loaded {total_frames} continuous authentic stereo frames from {base_dir}.")
    
    # Output Video Setup
    fps_out = 1.5  # 35 frames / 1.5 fps = 23.33 seconds duration (targets 20-30s)
    W, H = 1240, 609
    
    demo_video_path = os.path.join(demo_out_dir, "bussense_authentic_stereo_pothole_demo.mp4")
    debug_video_path = os.path.join(demo_out_dir, "bussense_authentic_stereo_debug.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_demo = cv2.VideoWriter(demo_video_path, fourcc, fps_out, (W, H))
    out_debug = cv2.VideoWriter(debug_video_path, fourcc, fps_out, (W, H))
    
    # Telemetry tracking
    t_start = time.time()
    total_potholes_detected = 0
    total_valid_potholes = 0
    total_cracks_detected = 0
    road_condition_transitions = []
    prev_condition = None
    
    # Temporal hysteresis state (smoothing condition)
    condition_history = []
    
    print("Beginning video rendering...")
    for idx, fname in enumerate(files):
        t_frame_start = time.time()
        fnum = int(os.path.splitext(fname)[0])
        img_l = cv2.imread(os.path.join(left_dir, fname))
        img_r = cv2.imread(os.path.join(right_dir, fname))
        
        # 1. PURE INFERENCE STEREO MATCHING
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disp_pred = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        
        valid_disp = disp_pred > 50.0
        depth_pred = np.zeros_like(disp_pred)
        depth_pred[valid_disp] = fB_m_px / disp_pred[valid_disp]
        
        # 2. POTHOLE SEGMENTATION
        res_p = pothole_model.predict(img_l, conf=0.15, verbose=False)[0]
        potholes = []
        
        vis_demo = img_l.copy()
        vis_cavity = np.zeros((H, W, 3), dtype=np.uint8)
        
        for i, box in enumerate(res_p.boxes):
            conf = float(box.conf[0])
            raw_mask = res_p.masks.data[i].cpu().numpy()
            mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask_pixels = int(np.sum(mask))
            if mask_pixels < 50:
                continue
                
            total_potholes_detected += 1
            
            # Local road ring (outer 35px, inner 11px)
            k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dil_out = cv2.dilate(mask.astype(np.uint8), k_out)
            dil_in = cv2.dilate(mask.astype(np.uint8), k_in)
            ring_mask = (dil_out > 0) & (dil_in == 0) & valid_disp
            
            valid_in_pixels = int(np.sum(mask & valid_disp))
            valid_frac = float(valid_in_pixels / max(mask_pixels, 1))
            
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                continue
            main_cnt = max(cnts, key=cv2.contourArea)
            
            # Check geometry validity
            if np.sum(ring_mask) < 40 or valid_in_pixels < 20 or valid_frac < 0.50:
                # Invalid geometry: keep track internally, but SUPPRESS visual overlay completely
                potholes.append({
                    "id": i, "valid": False, "conf": conf, "depth_cm": None, "area_m2": None
                })
                continue
                
            # Valid geometry: Fit plane
            v_ring, u_ring = np.where(ring_mask)
            d_ring = disp_pred[ring_mask]
            Z_ring = fB_m_px / d_ring
            X_ring = (u_ring - cx) * Z_ring / fx
            Y_ring = (v_ring - cy) * Z_ring / fy
            pts_ring = np.stack([X_ring, Y_ring, Z_ring], axis=1)
            normal, d_plane, centroid, plane_rmse_m = fit_plane_svd(pts_ring)
            
            # Inside cavity signed depth
            v_in, u_in = np.where(mask & valid_disp)
            d_in = disp_pred[mask & valid_disp]
            Z_in = fB_m_px / d_in
            X_in = (u_in - cx) * Z_in / fx
            Y_in = (v_in - cy) * Z_in / fy
            pts_in = np.stack([X_in, Y_in, Z_in], axis=1)
            
            perp_dists_cm = (np.dot(pts_in, normal) + d_plane) * 100.0
            p95_depth_cm = float(np.percentile(perp_dists_cm, 95))
            
            # Footprint area & dimensions
            pts_cnt_2d = main_cnt[:, 0, :]
            u_cnt = pts_cnt_2d[:, 0]
            v_cnt = pts_cnt_2d[:, 1]
            denom = normal[0] * ((u_cnt - cx) / fx) + normal[1] * ((v_cnt - cy) / fy) + normal[2]
            valid_rays = np.abs(denom) > 1e-4
            Z_cnt = -d_plane / denom[valid_rays]
            X_cnt = ((u_cnt[valid_rays] - cx) / fx) * Z_cnt
            Y_cnt = ((v_cnt[valid_rays] - cy) / fy) * Z_cnt
            pts_cnt_3d = np.stack([X_cnt, Y_cnt, Z_cnt], axis=1)
            
            area_m2, length_m, width_m = shoelace_and_dimensions_3d(pts_cnt_3d, normal)
            
            total_valid_potholes += 1
            potholes.append({
                "id": i, "valid": True, "conf": conf, "depth_cm": p95_depth_cm,
                "area_m2": area_m2, "length_m": length_m, "width_m": width_m
            })
            
            # Draw semi-transparent cavity fill
            overlay = vis_demo.copy()
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)
            vis_demo = overlay
            cv2.drawContours(vis_demo, [main_cnt], -1, (0, 0, 255), 2)
            
            # Draw bounding box & label
            x_c, y_c, w_c, h_c = cv2.boundingRect(main_cnt)
            label = f"POTHOLE #{i:02d} | Depth: {p95_depth_cm:.1f} cm | Area: {area_m2:.3f} m2"
            cv2.rectangle(vis_demo, (x_c, max(y_c - 24, 0)), (x_c + len(label)*9, max(y_c, 24)), (0, 0, 0), -1)
            cv2.putText(vis_demo, label, (x_c + 4, max(y_c - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            
            # Cavity depth visualization for debug
            dep_map = np.zeros((H, W), dtype=np.uint8)
            dep_map[mask & valid_disp] = np.clip(perp_dists_cm * 50.0, 0, 255).astype(np.uint8)
            dep_col = cv2.applyColorMap(dep_map, cv2.COLORMAP_MAGMA)
            vis_cavity[mask & valid_disp] = dep_col[mask & valid_disp]

        # 3. CRACK DETECTION
        res_c = crack_model.predict(img_l, conf=0.08, verbose=False)[0]
        cracks = []
        for j, cbox in enumerate(res_c.boxes):
            if int(cbox.cls[0]) != 1: continue
            c_conf = float(cbox.conf[0])
            c_raw = res_c.masks.data[j].cpu().numpy()
            c_mask = cv2.resize(c_raw, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if np.sum(c_mask) < 30: continue
            
            cnts_c, _ = cv2.findContours(c_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts_c) == 0: continue
            c_cnt = max(cnts_c, key=cv2.contourArea)
            pts_c_2d = c_cnt[:, 0, :]
            u_c = pts_c_2d[:, 0]
            v_c = pts_c_2d[:, 1]
            d_c = disp_pred[v_c, u_c]
            valid_c = d_c > 50.0
            if np.sum(valid_c) < 3: continue
            
            Z_c = fB_m_px / d_c[valid_c]
            X_c = ((u_c[valid_c] - cx) / fx) * Z_c
            Y_c = ((v_c[valid_c] - cy) / fy) * Z_c
            pts_c_3d = np.stack([X_c, Y_c, Z_c], axis=1)
            crack_len_m = float(np.sum(np.linalg.norm(np.diff(pts_c_3d, axis=0), axis=1)) * 0.5)
            
            total_cracks_detected += 1
            cracks.append({"id": j, "length_m": crack_len_m, "conf": c_conf})
            
            # Draw crack contour (neon cyan/yellow)
            cv2.drawContours(vis_demo, [c_cnt], -1, (0, 255, 255), 2)
            xc, yc, wc, hc = cv2.boundingRect(c_cnt)
            c_label = f"CRACK #{j:02d} | Length: {crack_len_m:.2f} m"
            cv2.putText(vis_demo, c_label, (xc, max(yc - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 4. ROAD CONDITION WITH HYSTERESIS
        valid_p_list = [p for p in potholes if p["valid"]]
        max_p_depth = max([p["depth_cm"] for p in valid_p_list], default=0.0)
        p_count = len(valid_p_list)
        c_count = len(cracks)
        
        if p_count == 0 and c_count == 0:
            raw_cond = "NORMAL"
        elif max_p_depth > 5.0 or p_count >= 3 or c_count >= 3:
            raw_cond = "ALARMING"
        elif max_p_depth > 2.0 or p_count >= 1 or c_count >= 2:
            raw_cond = "HIGH"
        elif p_count >= 1 or c_count >= 1:
            raw_cond = "MEDIUM"
        else:
            raw_cond = "NORMAL"
            
        condition_history.append(raw_cond)
        if len(condition_history) > 3:
            condition_history.pop(0)
            
        # Asymmetric hysteresis: escalate immediately, decay if sustained
        order = {"NORMAL": 0, "MEDIUM": 1, "HIGH": 2, "ALARMING": 3}
        inv_order = {0: "NORMAL", 1: "MEDIUM", 2: "HIGH", 3: "ALARMING"}
        smoothed_cond = inv_order[max([order[c] for c in condition_history])]
        
        if smoothed_cond != prev_condition:
            road_condition_transitions.append((fnum, prev_condition, smoothed_cond))
            prev_condition = smoothed_cond
            
        # 5. RENDER MAIN DEMO HUD OVERLAY
        # Frosted glass Top-Left Card (Traffic Telemetry)
        overlay_hud = vis_demo.copy()
        cv2.rectangle(overlay_hud, (15, 15), (380, 115), (20, 20, 20), -1)
        # Frosted glass Top-Right Card (Road Condition Telemetry)
        cv2.rectangle(overlay_hud, (W - 400, 15), (W - 15, 115), (20, 20, 20), -1)
        # Bottom-Left Summary Ticker
        cv2.rectangle(overlay_hud, (15, H - 75), (560, H - 15), (20, 20, 20), -1)
        # Bottom-Right Badge
        cv2.rectangle(overlay_hud, (W - 370, H - 55), (W - 15, H - 15), (20, 20, 20), -1)
        
        vis_demo = cv2.addWeighted(vis_demo, 0.25, overlay_hud, 0.75, 0)
        
        # Border glow for cards
        cv2.rectangle(vis_demo, (15, 15), (380, 115), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (W - 400, 15), (W - 15, 115), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (15, H - 75), (560, H - 15), (80, 80, 80), 1)
        cv2.rectangle(vis_demo, (W - 370, H - 55), (W - 15, H - 15), (80, 80, 80), 1)
        
        # Top-Left Content
        cv2.putText(vis_demo, "TRAFFIC: NORMAL", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(vis_demo, "VEHICLES: 0  |  DENSITY: 0 veh/km2", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        cv2.putText(vis_demo, "NEAREST VEHICLE: -- m", (25, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
        
        # Top-Right Content
        cond_color = (0, 255, 0) if smoothed_cond == "NORMAL" else (0, 255, 255) if smoothed_cond == "MEDIUM" else (0, 140, 255) if smoothed_cond == "HIGH" else (0, 0, 255)
        cv2.putText(vis_demo, f"ROAD CONDITION: {smoothed_cond}", (W - 385, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.68, cond_color, 2)
        cv2.putText(vis_demo, f"POTHOLES: {p_count:02d}  |  CRACKS: {c_count:02d}", (W - 385, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(vis_demo, f"FRAME {fnum:02d}/{total_frames:02d}  (Native Stereo)", (W - 385, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        
        # Bottom-Left Content (Pothole Metrology Details)
        if len(valid_p_list) > 0:
            first_p = valid_p_list[0]
            cv2.putText(vis_demo, f"POTHOLE #{first_p['id']:02d}: Depth: {first_p['depth_cm']:.1f} cm  |  Area: {first_p['area_m2']:.3f} m2",
                        (25, H - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(vis_demo, f"Length x Width: {first_p['length_m']:.2f} x {first_p['width_m']:.2f} m  |  Confidence: {first_p['conf']:.2f}",
                        (25, H - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)
        elif c_count > 0:
            first_c = cracks[0]
            cv2.putText(vis_demo, f"CRACK #{first_c['id']:02d}: Length: {first_c['length_m']:.2f} m  |  Confidence: {first_c['conf']:.2f}",
                        (25, H - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            cv2.putText(vis_demo, "NO ROAD DAMAGE DETECTED IN OBSERVABLE ROAD REGION", (25, H - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)
                        
        # Bottom-Right Content
        cv2.putText(vis_demo, "BusSense Unified Perception", (W - 355, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(vis_demo, "Native Stereo Metrology Rig", (W - 355, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        
        out_demo.write(vis_demo)
        
        # 6. RENDER DEBUG VIDEO (2x2 Quad View)
        # Q1: Left RGB + Detections (vis_demo without HUD for clarity)
        q1 = img_l.copy()
        for p in potholes:
            color = (0, 0, 255) if p["valid"] else (0, 0, 180)
            if p["valid"]:
                cv2.putText(q1, f"P#{p['id']}: {p['depth_cm']:.1f}cm, {p['area_m2']:.3f}m2", (30, 60 + 30*p['id']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        for c in cracks:
            cv2.putText(q1, f"C#{c['id']}: {c['length_m']:.2f}m", (30, 140 + 25*c['id']),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(q1, "Q1: LEFT RGB + DETECTIONS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Q2: Right RGB
        q2 = img_r.copy()
        cv2.putText(q2, "Q2: RIGHT RGB (NATIVE STEREO)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Q3: SGBM Disparity (Jet colormap)
        disp_norm = cv2.normalize(disp_pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        q3 = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        cv2.putText(q3, "Q3: SGBM DISPARITY MAP", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Q4: Metric 3D Depth Map + Cavity Overlay + Telemetry
        depth_norm = np.clip((depth_pred - 0.4) / 1.5 * 255.0, 0, 255).astype(np.uint8)
        q4 = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_TURBO)
        q4[~valid_disp] = 0
        
        # Overlay cavity depressions onto depth map
        cavity_mask = np.any(vis_cavity > 0, axis=-1)
        q4[cavity_mask] = vis_cavity[cavity_mask]
        
        # Telemetry HUD on Q4
        cv2.putText(q4, "Q4: METRIC 3D DEPTH & CAVITY METROLOGY", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(q4, f"ROAD: {smoothed_cond} | P: {p_count} | C: {c_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cond_color, 2)
        if len(valid_p_list) > 0:
            p0 = valid_p_list[0]
            cv2.putText(q4, f"P#{p0['id']}: Depth: {p0['depth_cm']:.1f}cm | Area: {p0['area_m2']:.3f}m2", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(q4, f"Dim: {p0['length_m']:.2f}m x {p0['width_m']:.2f}m", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
            
        # Downscale quadrants to fit into W x H (each quad is W//2 x H//2)
        qw, qh = W // 2, H // 2
        q1_s = cv2.resize(q1, (qw, qh))
        q2_s = cv2.resize(q2, (qw, qh))
        q3_s = cv2.resize(q3, (qw, qh))
        q4_s = cv2.resize(q4, (qw, qh))
        
        debug_frame = np.vstack([
            np.hstack([q1_s, q2_s]),
            np.hstack([q3_s, q4_s])
        ])
        # In case of odd pixels, resize back to exact W x H
        debug_frame = cv2.resize(debug_frame, (W, H))
        out_debug.write(debug_frame)
        
        t_frame_end = time.time()
        fps_frame = 1.0 / max(t_frame_end - t_frame_start, 1e-4)
        print(f"Rendered frame {idx+1:02d}/{total_frames:02d} (frame {fnum:02d}.png) in {t_frame_end - t_frame_start:.3f}s ({fps_frame:.2f} FPS)")
        
    out_demo.release()
    out_debug.release()
    t_total = time.time() - t_start
    actual_processing_fps = total_frames / t_total
    duration_s = total_frames / fps_out
    
    print("\n=== RENDERING COMPLETE ===")
    print(f"Main Demo: {demo_video_path} ({os.path.getsize(demo_video_path):,} bytes)")
    print(f"Debug Video: {debug_video_path} ({os.path.getsize(debug_video_path):,} bytes)")
    print(f"Total Frames: {total_frames}")
    print(f"Video Duration: {duration_s:.2f} seconds (Output FPS: {fps_out:.2f})")
    print(f"Total Processing Time: {t_total:.2f} seconds (Actual Processing FPS: {actual_processing_fps:.2f} FPS)")
    print(f"Total Potholes Detected: {total_potholes_detected}")
    print(f"Total Valid Pothole Measurements: {total_valid_potholes}")
    print(f"Total Cracks Detected: {total_cracks_detected}")
    print("Road Condition Transitions:")
    for tr in road_condition_transitions:
        print(f"  Frame {tr[0]:02d}: {tr[1]} -> {tr[2]}")

if __name__ == "__main__":
    main()
