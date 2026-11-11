import os
import sys
import json
import cv2
import numpy as np
from ultralytics import YOLO

def fit_plane_svd(pts: np.ndarray):
    """Fit 3D plane using SVD. Returns normal (a, b, c), d, centroid, and plane RMSE."""
    centroid = np.mean(pts, axis=0)
    _, s, vh = np.linalg.svd(pts - centroid)
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
      pts_2d: np.ndarray
    """
    if len(pts_3d) < 3:
        return 0.0, 0.0, 0.0, None
    # 2D orthonormal basis on plane
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
    
    # Minimum area bounding box in 2D plane coordinates
    pts_2d_cv = pts_2d.astype(np.float32)
    rect = cv2.minAreaRect(pts_2d_cv)
    w_box, h_box = rect[1]
    length = float(max(w_box, h_box))
    width = float(min(w_box, h_box))
    
    return float(area), length, width, pts_2d

def run_segmentation_metrology_validation():
    print("=== STARTING 10-FRAME AUTHENTIC STEREO SEGMENTATION & METROLOGY VALIDATION ===")
    
    base_dir = r"data\fan_road_surface\dataset_1"
    left_dir = os.path.join(base_dir, "left")
    right_dir = os.path.join(base_dir, "right")
    out_dir = r"outputs\evaluation"
    os.makedirs(out_dir, exist_ok=True)
    art_dir = r"C:\Users\Thrinay\.gemini\antigravity\brain\03d7fdef-cb64-4066-b840-f0b324ee084e"
    
    # Strict calibration isolation - Fan native stereo
    fx, fy = 698.804085, 698.804085
    cx, cy = 634.69008, 339.49033
    baseline_m = 0.119588
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
    
    # Representative 12 frames spanning clean road, cracks, and severe potholes
    eval_frames = [1, 4, 7, 10, 11, 12, 14, 15, 20, 22, 28, 34]
    
    results = []
    contact_sheet_paths = []
    
    total_potholes_found = 0
    total_valid_measurements = 0
    total_cracks_found = 0
    
    all_depths = []
    all_areas = []
    
    for fnum in eval_frames:
        fname = f"{fnum}.png"
        path_l = os.path.join(left_dir, fname)
        path_r = os.path.join(right_dir, fname)
        if not os.path.exists(path_l) or not os.path.exists(path_r):
            continue
            
        img_l = cv2.imread(path_l)
        img_r = cv2.imread(path_r)
        h, w = img_l.shape[:2]
        
        # PURE RUNTIME INFERENCE: No GT loaded!
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disp_pred = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        
        valid_disp = disp_pred > 50.0
        depth_pred = np.zeros_like(disp_pred)
        depth_pred[valid_disp] = fB_m_px / disp_pred[valid_disp]
        
        # Pothole detection on LEFT RGB
        res_p = pothole_model.predict(img_l, conf=0.15, verbose=False)[0]
        potholes_frame = []
        
        vis_mask_contour = img_l.copy()
        vis_cavity_depth = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i, box in enumerate(res_p.boxes):
            conf = float(box.conf[0])
            raw_mask = res_p.masks.data[i].cpu().numpy()
            mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            mask_pixels = int(np.sum(mask))
            if mask_pixels < 50:
                continue
                
            total_potholes_found += 1
            
            # Local road ring
            k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dil_out = cv2.dilate(mask.astype(np.uint8), k_out)
            dil_in = cv2.dilate(mask.astype(np.uint8), k_in)
            ring_mask = (dil_out > 0) & (dil_in == 0) & valid_disp
            
            valid_in_pixels = int(np.sum(mask & valid_disp))
            valid_stereo_frac = float(valid_in_pixels / max(mask_pixels, 1))
            
            # Draw contour and mask on vis
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (0, 0, 255) if i == 0 else (0, 165, 255)
            cv2.drawContours(vis_mask_contour, cnts, -1, color, 2)
            overlay = vis_mask_contour.copy()
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)
            vis_mask_contour = overlay
            
            # Guardrails for geometry validity
            if np.sum(ring_mask) < 40 or valid_in_pixels < 20 or valid_stereo_frac < 0.50:
                potholes_frame.append({
                    "pothole_id": i,
                    "confidence": round(conf, 3),
                    "mask_pixel_area": mask_pixels,
                    "valid_stereo_fraction": round(valid_stereo_frac, 3),
                    "status": "UNRELIABLE_GEOMETRY"
                })
                continue
                
            # Fit road plane to ring
            v_ring, u_ring = np.where(ring_mask)
            d_ring = disp_pred[ring_mask]
            Z_ring = fB_m_px / d_ring
            X_ring = (u_ring - cx) * Z_ring / fx
            Y_ring = (v_ring - cy) * Z_ring / fy
            pts_ring = np.stack([X_ring, Y_ring, Z_ring], axis=1)
            normal, d_plane, centroid, plane_rmse_m = fit_plane_svd(pts_ring)
            plane_rmse_mm = plane_rmse_m * 1000.0
            
            # Inside cavity 3D points
            v_in, u_in = np.where(mask & valid_disp)
            d_in = disp_pred[mask & valid_disp]
            Z_in = fB_m_px / d_in
            X_in = (u_in - cx) * Z_in / fx
            Y_in = (v_in - cy) * Z_in / fy
            pts_in = np.stack([X_in, Y_in, Z_in], axis=1)
            
            # Signed perpendicular depth (cm)
            perp_dists_cm = (np.dot(pts_in, normal) + d_plane) * 100.0
            p50_depth_cm = float(np.median(perp_dists_cm))
            p95_depth_cm = float(np.percentile(perp_dists_cm, 95))
            max_depth_cm = float(np.max(perp_dists_cm))
            
            # Cavity depth map visualization (depression depth > 0 is colored)
            dep_map = np.zeros((h, w), dtype=np.uint8)
            dep_map[mask & valid_disp] = np.clip(perp_dists_cm * 50.0, 0, 255).astype(np.uint8)
            dep_color = cv2.applyColorMap(dep_map, cv2.COLORMAP_MAGMA)
            vis_cavity_depth[mask & valid_disp] = dep_color[mask & valid_disp]
            
            # Ray-plane intersection for contour footprint
            main_cnt = max(cnts, key=cv2.contourArea)
            pts_cnt_2d = main_cnt[:, 0, :]
            u_cnt = pts_cnt_2d[:, 0]
            v_cnt = pts_cnt_2d[:, 1]
            denom = normal[0] * ((u_cnt - cx) / fx) + normal[1] * ((v_cnt - cy) / fy) + normal[2]
            valid_rays = np.abs(denom) > 1e-4
            
            Z_cnt = -d_plane / denom[valid_rays]
            X_cnt = ((u_cnt[valid_rays] - cx) / fx) * Z_cnt
            Y_cnt = ((v_cnt[valid_rays] - cy) / fy) * Z_cnt
            pts_cnt_3d = np.stack([X_cnt, Y_cnt, Z_cnt], axis=1)
            
            area_m2, length_m, width_m, _ = shoelace_and_dimensions_3d(pts_cnt_3d, normal)
            
            total_valid_measurements += 1
            all_depths.append(p95_depth_cm)
            all_areas.append(area_m2)
            
            potholes_frame.append({
                "pothole_id": i,
                "confidence": round(conf, 3),
                "mask_pixel_area": mask_pixels,
                "valid_stereo_fraction": round(valid_stereo_frac, 3),
                "plane_rmse_mm": round(plane_rmse_mm, 2),
                "physical_length_m": round(length_m, 3),
                "physical_width_m": round(width_m, 3),
                "footprint_area_m2": round(area_m2, 4),
                "signed_depth_cm": {
                    "p50": round(p50_depth_cm, 2),
                    "p95": round(p95_depth_cm, 2),
                    "max": round(max_depth_cm, 2)
                },
                "status": "VALID"
            })
            
            # Draw on vis
            cv2.putText(vis_mask_contour, f"P{i}: {p95_depth_cm:.1f}cm | {area_m2:.3f}m2",
                        (int(u_cnt[0]), int(v_cnt[0]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        # Crack detection on LEFT RGB
        res_c = crack_model.predict(img_l, conf=0.08, verbose=False)[0]
        cracks_frame = []
        for j, cbox in enumerate(res_c.boxes):
            if int(cbox.cls[0]) != 1: continue
            c_conf = float(cbox.conf[0])
            c_raw = res_c.masks.data[j].cpu().numpy()
            c_mask = cv2.resize(c_raw, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            c_pixels = int(np.sum(c_mask))
            if c_pixels < 30: continue
            
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
            
            pixel_width = float(c_pixels / max(len(c_cnt), 1))
            mean_Z = float(np.mean(Z_c))
            est_width_cm = float(pixel_width * (mean_Z / fx) * 100.0)
            
            total_cracks_found += 1
            cracks_frame.append({
                "crack_id": j,
                "confidence": round(c_conf, 3),
                "mask_pixels": c_pixels,
                "length_m": round(crack_len_m, 2),
                "estimated_surface_crack_width_cm": round(est_width_cm, 2)
            })
            cv2.drawContours(vis_mask_contour, [c_cnt], -1, (0, 255, 255), 2)
            
        # Road Condition Engine
        valid_potholes = [p for p in potholes_frame if p.get("status") == "VALID"]
        max_p_depth = max([p["signed_depth_cm"]["p95"] for p in valid_potholes], default=0.0)
        p_count = len(valid_potholes)
        c_count = len(cracks_frame)
        
        if p_count == 0 and c_count == 0:
            condition = "NORMAL"
        elif max_p_depth > 5.0 or p_count >= 3 or c_count >= 3:
            condition = "ALARMING"
        elif max_p_depth > 2.0 or p_count >= 1 or c_count >= 2:
            condition = "HIGH"
        elif p_count >= 1 or c_count >= 1:
            condition = "MEDIUM"
        else:
            condition = "NORMAL"
            
        frame_report = {
            "frame_num": fnum,
            "filename": fname,
            "potholes": potholes_frame,
            "cracks": cracks_frame,
            "road_condition": condition
        }
        results.append(frame_report)
        
        # Build 6-panel diagnostic sheet
        # Panel 1: Left RGB
        p1 = img_l.copy()
        cv2.putText(p1, f"Frame {fnum:02d} LEFT RGB", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Panel 2: Predicted Mask & Contour Overlay
        p2 = vis_mask_contour
        cv2.putText(p2, f"Pothole/Crack Segmentations ({len(valid_potholes)} P, {len(cracks_frame)} C)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Panel 3: SGBM Disparity Map
        disp_vis = cv2.normalize(disp_pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        p3 = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        cv2.putText(p3, "SGBM Disparity Map", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 4: Metric 3D Depth (meters)
        depth_vis = np.clip((depth_pred - 0.4) / 1.5 * 255.0, 0, 255).astype(np.uint8)
        p4 = cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_TURBO)
        p4[~valid_disp] = 0
        cv2.putText(p4, "Metric 3D Depth Map (m)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 5: Cavity Depression Heatmap (cm)
        p5 = vis_cavity_depth
        cv2.putText(p5, "Cavity Depression Depth (cm)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel 6: Telemetry HUD
        p6 = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(p6, f"FRAME {fnum:02d} METROLOGY TELEMETRY", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(p6, f"Source: Fan Native Stereo (1240x609)", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(p6, f"Intrinsics: fx=698.8, B=119.6mm", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        y_pos = 165
        for p in valid_potholes[:2]:
            cv2.putText(p6, f"P{p['pothole_id']}: Depth={p['signed_depth_cm']['p95']:.1f}cm | Area={p['footprint_area_m2']:.3f}m2",
                        (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(p6, f"    Dim: {p['physical_length_m']:.2f}m x {p['physical_width_m']:.2f}m | Ring RMSE={p['plane_rmse_mm']:.1f}mm",
                        (25, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)
            y_pos += 65
            
        for c in cracks_frame[:2]:
            cv2.putText(p6, f"Crack {c['crack_id']}: L={c['length_m']:.2f}m | Width={c['estimated_surface_crack_width_cm']:.1f}cm",
                        (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            y_pos += 35
            
        cv2.putText(p6, f"ROAD CONDITION: {condition}", (25, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        # Stitch 2x3 grid
        row1 = np.hstack([p1, p2, p3])
        row2 = np.hstack([p4, p5, p6])
        grid = np.vstack([row1, row2])
        grid_out = cv2.resize(grid, (1800, 600))
        
        sheet_name = f"authentic_seg_metrology_frame_{fnum:02d}.jpg"
        sheet_path = os.path.join(out_dir, sheet_name)
        cv2.imwrite(sheet_path, grid_out)
        cv2.imwrite(os.path.join(art_dir, sheet_name), grid_out)
        contact_sheet_paths.append(sheet_path)
        print(f"Processed Frame {fnum:02d}: {len(valid_potholes)} potholes, {len(cracks_frame)} cracks -> {condition}")

    summary = {
        "dataset": "Fan Native Stereo (data/fan_road_surface/dataset_1)",
        "calibration": {
            "resolution": "1240x609",
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "baseline_m": baseline_m,
            "fB": fB_m_px
        },
        "frames_evaluated_count": len(eval_frames),
        "total_potholes_detected": total_potholes_found,
        "total_valid_measurements": total_valid_measurements,
        "total_cracks_detected": total_cracks_found,
        "metrology_statistics": {
            "depth_p95_cm": {
                "min": round(float(np.min(all_depths)), 2) if all_depths else 0.0,
                "median": round(float(np.median(all_depths)), 2) if all_depths else 0.0,
                "mean": round(float(np.mean(all_depths)), 2) if all_depths else 0.0,
                "max": round(float(np.max(all_depths)), 2) if all_depths else 0.0
            },
            "footprint_area_m2": {
                "min": round(float(np.min(all_areas)), 4) if all_areas else 0.0,
                "median": round(float(np.median(all_areas)), 4) if all_areas else 0.0,
                "mean": round(float(np.mean(all_areas)), 4) if all_areas else 0.0,
                "max": round(float(np.max(all_areas)), 4) if all_areas else 0.0
            }
        },
        "frame_details": results,
        "contact_sheets": contact_sheet_paths
    }
    
    out_json = os.path.join(out_dir, "authentic_pothole_segmentation_metrology.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved: {out_json}")

if __name__ == "__main__":
    run_segmentation_metrology_validation()
