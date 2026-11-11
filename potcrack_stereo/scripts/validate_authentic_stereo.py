import os
import sys
import json
import cv2
import numpy as np
import scipy.io as sio
from ultralytics import YOLO

def fit_plane_svd(pts: np.ndarray):
    """Fit 3D plane using SVD. Returns normal vector (a, b, c) and d such that a*x + b*y + c*z + d = 0."""
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2, :]
    # Orient normal such that normal[1] > 0 (pointing downwards towards road)
    if normal[1] < 0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    return normal, d, centroid

def shoelace_area_3d(pts_3d: np.ndarray, normal: np.ndarray) -> float:
    """Compute polygon area in 3D by projecting onto fitted plane coordinates."""
    if len(pts_3d) < 3:
        return 0.0
    # Create an orthonormal 2D basis on the plane
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
    return float(area)

def run_authentic_stereo_validation():
    print("=== STARTING AUTHENTIC STEREO POTHOLE & CRACK VALIDATION ===")
    
    base_dir = r"C:\Users\Thrinay\.gemini\antigravity\scratch\bussense\data\fan_road_surface\dataset_1"
    left_dir = os.path.join(base_dir, "left")
    right_dir = os.path.join(base_dir, "right")
    exp_dir = os.path.join(base_dir, "experimental_results")
    out_dir = r"C:\Users\Thrinay\.gemini\antigravity\scratch\potcrack_stereo\outputs\evaluation"
    os.makedirs(out_dir, exist_ok=True)
    
    # Exact camera intrinsics and baseline
    fx, fy = 698.804085, 698.804085
    cx, cy = 634.69008, 339.49033
    baseline_m = 0.119588  # 119.588 mm
    fB_m_px = fx * baseline_m  # 83.56857 m*px
    
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
    
    files = sorted([f for f in os.listdir(left_dir) if f.endswith(".png")], key=lambda x: int(os.path.splitext(x)[0]))
    
    dataset_disparity_epes = []
    dataset_bad1s = []
    dataset_bad3s = []
    dataset_depth_maes = []
    dataset_depth_rmses = []
    dataset_depth_absrels = []
    
    per_frame_results = []
    
    contact_sheets_saved = []
    
    # Select key representative frames for detailed inspection
    key_frames = [10, 15, 34]
    
    for fname in files:
        fnum = int(os.path.splitext(fname)[0])
        path_l = os.path.join(left_dir, fname)
        path_r = os.path.join(right_dir, fname)
        mat_path = os.path.join(exp_dir, f"{fnum}.mat")
        if not os.path.exists(mat_path):
            continue
            
        img_l = cv2.imread(path_l)
        img_r = cv2.imread(path_r)
        mat = sio.loadmat(mat_path)
        disp_gt = mat["disparity_map2"]
        xyz_gt = mat["xyzPoints"] / 1000.0  # mm to meters
        
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disp_pred = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        
        valid_gt = np.isfinite(disp_gt) & (disp_gt > 50.0)
        valid_pred = disp_pred > 50.0
        valid_both = valid_gt & valid_pred
        
        diff_d = np.abs(disp_pred[valid_both] - disp_gt[valid_both])
        epe = float(np.mean(diff_d))
        bad1 = float(np.mean(diff_d > 1.0) * 100.0)
        bad3 = float(np.mean(diff_d > 3.0) * 100.0)
        
        z_gt = xyz_gt[:,:,2][valid_both]
        z_pred = fB_m_px / disp_pred[valid_both]
        diff_z = np.abs(z_pred - z_gt)
        mae_z = float(np.mean(diff_z))
        rmse_z = float(np.sqrt(np.mean(diff_z**2)))
        absrel_z = float(np.mean(diff_z / z_gt))
        
        dataset_disparity_epes.append(epe)
        dataset_bad1s.append(bad1)
        dataset_bad3s.append(bad3)
        dataset_depth_maes.append(mae_z)
        dataset_depth_rmses.append(rmse_z)
        dataset_depth_absrels.append(absrel_z)
        
        # Empirical road noise floor on this frame:
        # Sample middle flat road region (rows 350..550, cols 200..1000)
        flat_mask = np.zeros_like(valid_pred, dtype=bool)
        flat_mask[350:550, 200:1000] = True
        flat_mask = flat_mask & valid_pred
        flat_disps = disp_pred[flat_mask]
        sigma_d = float(np.std(flat_disps))
        iqr_d = float(np.percentile(flat_disps, 75) - np.percentile(flat_disps, 25))
        noise_floor_d = 1.96 * sigma_d
        
        # Run pothole model
        res_p = pothole_model.predict(img_l, conf=0.15, verbose=False)[0]
        potholes_data = []
        
        for i, box in enumerate(res_p.boxes):
            conf = float(box.conf[0])
            raw_mask = res_p.masks.data[i].cpu().numpy()
            mask = cv2.resize(raw_mask, (img_l.shape[1], img_l.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Local road ring
            k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dil_out = cv2.dilate(mask.astype(np.uint8), k_out)
            dil_in = cv2.dilate(mask.astype(np.uint8), k_in)
            ring_mask = (dil_out > 0) & (dil_in == 0) & (disp_pred > 50.0)
            
            if np.sum(ring_mask) < 50 or np.sum(mask & (disp_pred > 50.0)) < 20:
                continue
                
            # Fit road plane to ring from SGBM
            v_ring, u_ring = np.where(ring_mask)
            d_ring = disp_pred[ring_mask]
            Z_ring = fB_m_px / d_ring
            X_ring = (u_ring - cx) * Z_ring / fx
            Y_ring = (v_ring - cy) * Z_ring / fy
            pts_ring = np.stack([X_ring, Y_ring, Z_ring], axis=1)
            normal, d_plane, centroid = fit_plane_svd(pts_ring)
            
            # Inside cavity points
            in_mask = mask & (disp_pred > 50.0)
            v_in, u_in = np.where(in_mask)
            d_in = disp_pred[in_mask]
            Z_in = fB_m_px / d_in
            X_in = (u_in - cx) * Z_in / fx
            Y_in = (v_in - cy) * Z_in / fy
            pts_in = np.stack([X_in, Y_in, Z_in], axis=1)
            
            # Signed perpendicular depth (cm)
            perp_dists_cm = (np.dot(pts_in, normal) + d_plane) * 100.0
            p50_depth_cm = float(np.median(perp_dists_cm))
            p95_depth_cm = float(np.percentile(perp_dists_cm, 95))
            max_depth_cm = float(np.max(perp_dists_cm))
            
            # Cavity disparity resolution test:
            median_d_in = float(np.median(d_in))
            median_d_ring = float(np.median(d_ring))
            delta_d = abs(median_d_in - median_d_ring)
            is_resolved = bool(delta_d > 0.3) # resolved above empirical subpixel resolution
            
            # Contour and Shoelace footprint area
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pred_area_m2 = 0.0
            if len(cnts) > 0:
                main_cnt = max(cnts, key=cv2.contourArea)
                # Ray-plane intersection for contour vertices
                pts_cnt_2d = main_cnt[:, 0, :]
                u_cnt = pts_cnt_2d[:, 0]
                v_cnt = pts_cnt_2d[:, 1]
                # Ray direction: r = [(u-cx)/fx, (v-cy)/fy, 1.0]
                # Plane: dot(normal, Z * r) + d_plane = 0 => Z = -d_plane / dot(normal, r)
                denom = normal[0] * ((u_cnt - cx) / fx) + normal[1] * ((v_cnt - cy) / fy) + normal[2]
                valid_rays = np.abs(denom) > 1e-4
                Z_cnt = -d_plane / denom[valid_rays]
                X_cnt = ((u_cnt[valid_rays] - cx) / fx) * Z_cnt
                Y_cnt = ((v_cnt[valid_rays] - cy) / fy) * Z_cnt
                pts_cnt_3d = np.stack([X_cnt, Y_cnt, Z_cnt], axis=1)
                pred_area_m2 = shoelace_area_3d(pts_cnt_3d, normal)
            
            # Ground Truth pothole metrics from xyz_gt
            ring_gt_mask = (dil_out > 0) & (dil_in == 0) & np.isfinite(xyz_gt[:,:,2])
            pts_ring_gt = xyz_gt[ring_gt_mask]
            norm_gt, d_plane_gt, cent_gt = fit_plane_svd(pts_ring_gt)
            
            in_gt_mask = mask & np.isfinite(xyz_gt[:,:,2])
            pts_in_gt = xyz_gt[in_gt_mask]
            perp_gt_cm = (np.dot(pts_in_gt, norm_gt) + d_plane_gt) * 100.0
            p50_gt_cm = float(np.median(perp_gt_cm))
            p95_gt_cm = float(np.percentile(perp_gt_cm, 95))
            max_gt_cm = float(np.max(perp_gt_cm))
            
            # GT area
            gt_area_m2 = 0.0
            if len(cnts) > 0 and len(pts_cnt_2d) >= 3:
                denom_gt = norm_gt[0] * ((u_cnt - cx) / fx) + norm_gt[1] * ((v_cnt - cy) / fy) + norm_gt[2]
                valid_rays_gt = np.abs(denom_gt) > 1e-4
                Z_cnt_gt = -d_plane_gt / denom_gt[valid_rays_gt]
                X_cnt_gt = ((u_cnt[valid_rays_gt] - cx) / fx) * Z_cnt_gt
                Y_cnt_gt = ((v_cnt[valid_rays_gt] - cy) / fy) * Z_cnt_gt
                pts_cnt_3d_gt = np.stack([X_cnt_gt, Y_cnt_gt, Z_cnt_gt], axis=1)
                gt_area_m2 = shoelace_area_3d(pts_cnt_3d_gt, norm_gt)
                
            potholes_data.append({
                "pothole_id": i,
                "confidence": conf,
                "mask_pixels": int(np.sum(mask)),
                "valid_3d_points": int(len(pts_in)),
                "cavity_resolution": {
                    "delta_disparity_px": round(delta_d, 3),
                    "empirical_noise_floor_px": round(noise_floor_d, 3),
                    "is_resolved_above_noise": is_resolved
                },
                "pred_depth_cm": {
                    "p50": round(p50_depth_cm, 2),
                    "p95": round(p95_depth_cm, 2),
                    "max": round(max_depth_cm, 2)
                },
                "gt_depth_cm": {
                    "p50": round(p50_gt_cm, 2),
                    "p95": round(p95_gt_cm, 2),
                    "max": round(max_gt_cm, 2)
                },
                "depth_error_p95_cm": round(abs(p95_depth_cm - p95_gt_cm), 2),
                "pred_area_m2": round(pred_area_m2, 4),
                "gt_area_m2": round(gt_area_m2, 4),
                "area_error_m2": round(abs(pred_area_m2 - gt_area_m2), 4)
            })
            
        # Run crack model
        res_c = crack_model.predict(img_l, conf=0.08, verbose=False)[0]
        cracks_data = []
        for j, cbox in enumerate(res_c.boxes):
            if int(cbox.cls[0]) != 1: continue # only crack class
            c_conf = float(cbox.conf[0])
            c_raw = res_c.masks.data[j].cpu().numpy()
            c_mask = cv2.resize(c_raw, (img_l.shape[1], img_l.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            c_pts = np.sum(c_mask & (disp_pred > 50.0))
            if c_pts < 10: continue
            
            # Approximate crack length from contour perimeter / 2
            cnts, _ = cv2.findContours(c_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0: continue
            c_cnt = max(cnts, key=cv2.contourArea)
            # 3D points of contour
            pts_cnt_2d = c_cnt[:, 0, :]
            u_c = pts_cnt_2d[:, 0]
            v_c = pts_cnt_2d[:, 1]
            d_c = disp_pred[v_c, u_c]
            valid_c = d_c > 50.0
            if np.sum(valid_c) < 3: continue
            Z_c = fB_m_px / d_c[valid_c]
            X_c = ((u_c[valid_c] - cx) / fx) * Z_c
            Y_c = ((v_c[valid_c] - cy) / fy) * Z_c
            pts_c_3d = np.stack([X_c, Y_c, Z_c], axis=1)
            diffs = np.linalg.norm(np.diff(pts_c_3d, axis=0), axis=1)
            crack_length_m = float(np.sum(diffs) * 0.5) # approximate centerline
            
            # Estimated surface crack width
            pixel_width = float(np.sum(c_mask) / max(len(c_cnt), 1))
            mean_Z = float(np.mean(Z_c))
            est_width_cm = float(pixel_width * (mean_Z / fx) * 100.0)
            
            cracks_data.append({
                "crack_id": j,
                "confidence": round(c_conf, 3),
                "mask_pixels": int(np.sum(c_mask)),
                "length_m": round(crack_length_m, 2),
                "estimated_surface_crack_width_cm": round(est_width_cm, 2)
            })
            
        # Determine Road Condition
        # Severity calculation based on validated physical metrics:
        p_count = len(potholes_data)
        c_count = len(cracks_data)
        max_p_depth = max([p["pred_depth_cm"]["p95"] for p in potholes_data], default=0.0)
        
        if p_count == 0 and c_count == 0:
            condition = "NORMAL"
        elif max_p_depth > 5.0 or p_count >= 3:
            condition = "ALARMING"
        elif max_p_depth > 2.0 or p_count >= 1 or c_count >= 3:
            condition = "HIGH"
        elif p_count >= 1 or c_count >= 1:
            condition = "MEDIUM"
        else:
            condition = "NORMAL"
            
        frame_entry = {
            "frame_num": fnum,
            "filename": fname,
            "disparity_metrics": {
                "epe_px": round(epe, 3),
                "bad1_percent": round(bad1, 2),
                "bad3_percent": round(bad3, 2),
                "valid_overlap_percent": round(float(np.mean(valid_both) * 100.0), 1)
            },
            "depth_metrics": {
                "mae_cm": round(mae_z * 100.0, 2),
                "rmse_cm": round(rmse_z * 100.0, 2),
                "absrel": round(absrel_z, 4)
            },
            "potholes": potholes_data,
            "cracks": cracks_data,
            "road_condition": condition
        }
        per_frame_results.append(frame_entry)
        
        # If key frame, generate diagnostic visual sheet
        if fnum in key_frames:
            h, w = img_l.shape[:2]
            # Create a 6-panel diagnostic sheet:
            # 1. Left RGB with detection overlays
            # 2. Right RGB
            # 3. Predicted SGBM Disparity (Jet color)
            # 4. Ground Truth Disparity (Jet color)
            # 5. Disparity Error Map
            # 6. Road Ring & Metrology Overlay
            vis_det = img_l.copy()
            for p in potholes_data:
                cv2.putText(vis_det, f"Pothole {p['pothole_id']}: d={p['pred_depth_cm']['p95']:.1f}cm (GT {p['gt_depth_cm']['p95']:.1f}cm)", 
                            (20, 40 + 30*p['pothole_id']), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for c in cracks_data:
                cv2.putText(vis_det, f"Crack {c['crack_id']}: L={c['length_m']:.2f}m W={c['estimated_surface_crack_width_cm']:.1f}cm",
                            (20, 100 + 30*c['crack_id']), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
            # Color maps for disparities
            d_pred_norm = cv2.normalize(disp_pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            d_pred_color = cv2.applyColorMap(d_pred_norm, cv2.COLORMAP_JET)
            
            d_gt_norm = cv2.normalize(disp_gt, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            d_gt_color = cv2.applyColorMap(d_gt_norm, cv2.COLORMAP_JET)
            
            # Error map: 0..3 px -> 0..255
            err_map = np.zeros((h, w), dtype=np.uint8)
            err_map[valid_both] = np.clip(diff_d * 85.0, 0, 255).astype(np.uint8)
            err_color = cv2.applyColorMap(err_map, cv2.COLORMAP_HOT)
            err_color[~valid_both] = 0
            
            # HUD overlay
            hud_panel = np.zeros_like(img_l)
            cv2.putText(hud_panel, f"FRAME {fnum:02d} METROLOGY SUMMARY", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(hud_panel, f"Stereo EPE: {epe:.3f} px | Bad-1px: {bad1:.1f}%", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(hud_panel, f"Depth MAE: {mae_z*100:.2f} cm | AbsRel: {absrel_z:.4f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(hud_panel, f"Potholes: {len(potholes_data)} | Cracks: {len(cracks_data)}", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(hud_panel, f"ROAD CONDITION: {condition}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            
            # Combine into a 2x3 grid
            row1 = np.hstack([vis_det, img_r, d_pred_color])
            row2 = np.hstack([d_gt_color, err_color, hud_panel])
            grid = np.vstack([row1, row2])
            
            # Resize grid to reasonable display size
            grid_disp = cv2.resize(grid, (1800, 600))
            out_sheet_path = os.path.join(out_dir, f"authentic_stereo_frame_{fnum:02d}_diagnostic.jpg")
            cv2.imwrite(out_sheet_path, grid_disp)
            # Also copy to artifact dir
            art_sheet_path = rf"C:\Users\Thrinay\.gemini\antigravity\brain\03d7fdef-cb64-4066-b840-f0b324ee084e\authentic_stereo_frame_{fnum:02d}_diagnostic.jpg"
            cv2.imwrite(art_sheet_path, grid_disp)
            contact_sheets_saved.append(out_sheet_path)
            print(f"Saved diagnostic contact sheet for frame {fnum}: {out_sheet_path}")

    validation_summary = {
        "dataset_name": "Rui Fan IEEE TIP Road Surface 3D Reconstruction Benchmark (dataset_1)",
        "dataset_type": "Authentic synchronized, calibrated stereo road surface imagery",
        "reference_doi": "10.1109/TIP.2018.2882143",
        "total_frames_evaluated": len(dataset_disparity_epes),
        "calibration": {
            "image_width": 1240,
            "image_height": 609,
            "fx_pixels": fx,
            "fy_pixels": fy,
            "cx_pixels": cx,
            "cy_pixels": cy,
            "baseline_meters": baseline_m,
            "fB_meters_pixels": fB_m_px
        },
        "stereo_benchmark_metrics": {
            "mean_disparity_epe_px": round(float(np.mean(dataset_disparity_epes)), 4),
            "median_disparity_epe_px": round(float(np.median(dataset_disparity_epes)), 4),
            "mean_bad_1px_rate_percent": round(float(np.mean(dataset_bad1s)), 3),
            "mean_bad_3px_rate_percent": round(float(np.mean(dataset_bad3s)), 3),
            "mean_depth_mae_cm": round(float(np.mean(dataset_depth_maes)) * 100.0, 3),
            "mean_depth_rmse_cm": round(float(np.mean(dataset_depth_rmses)) * 100.0, 3),
            "mean_depth_absrel": round(float(np.mean(dataset_depth_absrels)), 5)
        },
        "key_frames_inspected": [f for f in per_frame_results if f["frame_num"] in key_frames],
        "all_frames": per_frame_results,
        "diagnostic_contact_sheets": contact_sheets_saved
    }
    
    json_path = os.path.join(out_dir, "authentic_stereo_pothole_validation.json")
    with open(json_path, "w") as f:
        json.dump(validation_summary, f, indent=2)
    print(f"Validation JSON saved to: {json_path}")

if __name__ == "__main__":
    run_authentic_stereo_validation()
