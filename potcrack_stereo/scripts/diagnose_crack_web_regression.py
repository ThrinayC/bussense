#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnose_crack_web_regression.py
========================================
Comprehensive Root-Cause Diagnostic for Crack-Branch Failure on Web/Alligator Cracks:
- Step 1: RAW model output for class 1 = crack across conf thresholds (0.03, 0.05, 0.08, 0.10, 0.15)
- Step 2: Inference resolution impact (640 vs 960 vs 1280 vs Road-ROI / Tiled)
- Step 3: Spatial gating pipeline step-by-step audit
- Step 4: Crack-specific skeletonization & 3D road-plane ray-intersection length
- Step 5: Root cause determination (A, B, C, D, or E) + diagnostic outputs
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.crack_geometry import CrackGeometryEngine, zhang_suen_thinning
from scripts.pothole_geometry import PotholeGeometryEngine


def run_raw_inference(
    model: YOLO,
    img: np.ndarray,
    imgsz: int = 640,
    conf_thresh: float = 0.01
) -> List[Dict[str, Any]]:
    """Runs raw YOLO inference and extracts all class 1 (crack) detections."""
    H, W = img.shape[:2]
    res = model.predict(img, imgsz=imgsz, conf=conf_thresh, verbose=False)[0]
    cracks = []
    if res.boxes is not None and res.masks is not None and len(res.boxes) > 0:
        boxes = res.boxes
        masks = res.masks.data.cpu().numpy()
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            if cls_id != 1 and len(model.names) > 1:
                continue
            conf = float(box.conf[0].item())
            xyxy = [int(v) for v in box.xyxy[0].cpu().numpy()]
            m_raw = masks[i]
            m_resized = cv2.resize((m_raw > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            nz = int(np.sum(m_resized))
            if nz == 0:
                continue
            cracks.append({
                "idx": i,
                "confidence": conf,
                "bbox": xyxy,
                "mask": m_resized,
                "pixel_area": nz,
                "center": [(xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2]
            })
    return cracks


def run_tiled_inference(
    model: YOLO,
    img: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 160,
    conf_thresh: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Runs tiled inference over the road region to preserve thin 1-pixel crack features
    without downsampling loss.
    """
    H, W = img.shape[:2]
    # Restrict tiling to the observable road region (lower 65% of 1080p image: y from 378 to 1080)
    y_start = int(0.35 * H)
    road_crop = img[y_start:H, 0:W]
    rH, rW = road_crop.shape[:2]

    # Two horizontal tiles covering 1920 width with overlap
    # Tile 1: x in [0, 1080], Tile 2: x in [840, 1920]
    tiles = [
        (0, 0, 1080, rH),
        (840, 0, W, rH)
    ]

    all_cracks = []
    for tx1, ty1, tx2, ty2 in tiles:
        tile_img = road_crop[ty1:ty2, tx1:tx2]
        res = model.predict(tile_img, imgsz=1024, conf=conf_thresh, verbose=False)[0]
        if res.boxes is not None and res.masks is not None and len(res.boxes) > 0:
            boxes = res.boxes
            masks = res.masks.data.cpu().numpy()
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                if cls_id != 1 and len(model.names) > 1:
                    continue
                conf = float(box.conf[0].item())
                b_xyxy = box.xyxy[0].cpu().numpy()
                # Global coordinates
                gx1 = int(b_xyxy[0]) + tx1
                gy1 = int(b_xyxy[1]) + ty1 + y_start
                gx2 = int(b_xyxy[2]) + tx1
                gy2 = int(b_xyxy[3]) + ty1 + y_start

                m_tile = cv2.resize((masks[i] > 0.5).astype(np.uint8), (tx2 - tx1, ty2 - ty1), interpolation=cv2.INTER_NEAREST)
                full_m = np.zeros((H, W), dtype=np.uint8)
                full_m[ty1 + y_start : ty2 + y_start, tx1 : tx2] = m_tile
                nz = int(np.sum(full_m))
                if nz == 0:
                    continue
                all_cracks.append({
                    "confidence": conf,
                    "bbox": [gx1, gy1, gx2, gy2],
                    "mask": full_m,
                    "pixel_area": nz,
                    "center": [(gx1 + gx2) // 2, (gy1 + gy2) // 2]
                })

    # Deduplicate overlapping detections via IoU / Mask overlap
    dedup = []
    all_cracks.sort(key=lambda x: x["confidence"], reverse=True)
    for c in all_cracks:
        duplicate = False
        for d in dedup:
            inter = cv2.bitwise_and(c["mask"], d["mask"])
            inter_nz = int(np.sum(inter))
            min_nz = min(c["pixel_area"], d["pixel_area"])
            if min_nz > 0 and (inter_nz / min_nz) > 0.40:
                duplicate = True
                break
        if not duplicate:
            dedup.append(c)
    return dedup


def main():
    print("============================================================")
    print("CRACK-BRANCH FAILURE INVESTIGATION ON WEB/ALLIGATOR CRACKS")
    print("============================================================")

    out_dir = PROJECT_ROOT / "outputs" / "debug" / "crack_web_regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    left_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "left"
    right_dir = PROJECT_ROOT / "data" / "pothole_long_stereo" / "right"

    model_path = PROJECT_ROOT / "checkpoints" / "segmentation" / "combined_pothole_crack" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    print(f"Loaded combined model: {model_path} (classes: {model.names})")

    calib = {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 960.0,
        "cy": 540.0,
        "baseline": 0.12
    }

    # Stereo Matcher for computing road plane
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=7,
        P1=8 * 3 * 7**2,
        P2=32 * 3 * 7**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    crack_engine = CrackGeometryEngine()

    # Representative test frames
    # 00: Control (clean pavement)
    # 22: Minor cracking
    # 41, 44, 45, 49: Sequence 1 with obvious alligator / web cracking
    # 63, 65, 66: Sequence 2 with severe deep alligator / web cracking
    target_frames = [0, 22, 41, 44, 45, 49, 63, 65, 66]

    summary_table = []
    full_json_records = {}

    thresholds = [0.03, 0.05, 0.08, 0.10, 0.15]

    for f_idx in target_frames:
        f_name = f"frame_{f_idx:05d}.jpg"
        img_path = left_dir / f_name
        r_path = right_dir / f_name
        img = cv2.imread(str(img_path))
        r_img = cv2.imread(str(r_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        print(f"\n------------------------------------------------------------")
        print(f"DIAGNOSING FRAME {f_idx:02d} ({f_name})")
        print(f"------------------------------------------------------------")

        # -------------------------------------------------------------
        # STEP 1: RAW MODEL OUTPUT & THRESHOLD SWEEP (imgsz=640)
        # -------------------------------------------------------------
        raw_cracks_640 = run_raw_inference(model, img, imgsz=640, conf_thresh=0.01)
        confs_640 = [c["confidence"] for c in raw_cracks_640]
        max_conf_640 = float(max(confs_640)) if confs_640 else 0.0

        cnt_03 = sum(1 for c in confs_640 if c >= 0.03)
        cnt_05 = sum(1 for c in confs_640 if c >= 0.05)
        cnt_08 = sum(1 for c in confs_640 if c >= 0.08)
        cnt_10 = sum(1 for c in confs_640 if c >= 0.10)
        cnt_15 = sum(1 for c in confs_640 if c >= 0.15)

        print(f"  STEP 1 (RAW imgsz=640): Total={len(raw_cracks_640)}, MaxConf={max_conf_640:.3f}")
        print(f"    >=0.03: {cnt_03} | >=0.05: {cnt_05} | >=0.08: {cnt_08} | >=0.10: {cnt_10} | >=0.15: {cnt_15}")

        # Save RAW debug image at conf >= 0.03 showing raw mask coverage
        vis_raw = img.copy()
        raw_mask_union = np.zeros((H, W), dtype=np.uint8)
        for c in raw_cracks_640:
            if c["confidence"] >= 0.03:
                raw_mask_union = cv2.bitwise_or(raw_mask_union, c["mask"] * 255)
                bx = c["bbox"]
                cv2.rectangle(vis_raw, (bx[0], bx[1]), (bx[2], bx[3]), (0, 165, 255), 1)
                cv2.putText(vis_raw, f"{c['confidence']:.2f}", (bx[0], max(bx[1]-3, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        overlay_raw = vis_raw.copy()
        overlay_raw[raw_mask_union > 0] = (0.3 * overlay_raw[raw_mask_union > 0] + 0.7 * np.array([0, 255, 255])).astype(np.uint8)
        vis_raw = overlay_raw
        cv2.putText(vis_raw, f"RAW CRACK DETECTIONS (conf>=0.03, count={cnt_03}, max={max_conf_640:.2f})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"step1_raw_conf03_frame_{f_idx:02d}.jpg"), vis_raw)

        # -------------------------------------------------------------
        # STEP 2: RESOLUTION & ROAD-ROI / TILED INFERENCE
        # -------------------------------------------------------------
        raw_cracks_960 = run_raw_inference(model, img, imgsz=960, conf_thresh=0.01)
        raw_cracks_1280 = run_raw_inference(model, img, imgsz=1280, conf_thresh=0.01)
        raw_cracks_tiled = run_tiled_inference(model, img, tile_size=1024, conf_thresh=0.01)

        confs_960 = [c["confidence"] for c in raw_cracks_960]
        confs_1280 = [c["confidence"] for c in raw_cracks_1280]
        confs_tiled = [c["confidence"] for c in raw_cracks_tiled]

        max_conf_960 = float(max(confs_960)) if confs_960 else 0.0
        max_conf_1280 = float(max(confs_1280)) if confs_1280 else 0.0
        max_conf_tiled = float(max(confs_tiled)) if confs_tiled else 0.0

        cnt_tiled_03 = sum(1 for c in confs_tiled if c >= 0.03)
        cnt_tiled_05 = sum(1 for c in confs_tiled if c >= 0.05)
        cnt_tiled_08 = sum(1 for c in confs_tiled if c >= 0.08)

        print(f"  STEP 2 (RESOLUTION COMPARISON):")
        print(f"    imgsz=640 : raw={len(raw_cracks_640):3d}, max_conf={max_conf_640:.3f}, >=0.05: {cnt_05:2d}, >=0.08: {cnt_08:2d}")
        print(f"    imgsz=960 : raw={len(raw_cracks_960):3d}, max_conf={max_conf_960:.3f}, >=0.05: {sum(1 for c in confs_960 if c>=0.05):2d}, >=0.08: {sum(1 for c in confs_960 if c>=0.08):2d}")
        print(f"    imgsz=1280: raw={len(raw_cracks_1280):3d}, max_conf={max_conf_1280:.3f}, >=0.05: {sum(1 for c in confs_1280 if c>=0.05):2d}, >=0.08: {sum(1 for c in confs_1280 if c>=0.08):2d}")
        print(f"    TILED-1024: raw={len(raw_cracks_tiled):3d}, max_conf={max_conf_tiled:.3f}, >=0.05: {cnt_tiled_05:2d}, >=0.08: {cnt_tiled_08:2d}")

        # Save resolution comparison image for representative frames
        vis_res_comp = np.hstack([
            cv2.resize(vis_raw, (960, 540)),
            cv2.resize(img, (960, 540))
        ])
        # Draw tiled results on right side
        vis_tiled = img.copy()
        tiled_mask_union = np.zeros((H, W), dtype=np.uint8)
        for c in raw_cracks_tiled:
            if c["confidence"] >= 0.03:
                tiled_mask_union = cv2.bitwise_or(tiled_mask_union, c["mask"] * 255)
        vis_tiled[tiled_mask_union > 0] = (0.3 * vis_tiled[tiled_mask_union > 0] + 0.7 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.putText(vis_tiled, f"TILED 1024 (conf>=0.03, count={cnt_tiled_03}, max={max_conf_tiled:.2f})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        vis_comp_pair = np.vstack([
            vis_raw,
            vis_tiled
        ])
        vis_comp_pair_s = cv2.resize(vis_comp_pair, (1280, 1440))
        cv2.imwrite(str(out_dir / f"step2_res_tiled_compare_frame_{f_idx:02d}.jpg"), vis_comp_pair_s)

        # -------------------------------------------------------------
        # STEP 3: SPATIAL GATING AUDIT
        # -------------------------------------------------------------
        # Trace where detections are eliminated:
        # Candidate set: raw_cracks_640 (at conf >= 0.03)
        stage_raw = [c for c in raw_cracks_640 if c["confidence"] >= 0.03]

        # Gating Check 1: Macro-scene artifact (>30% of screen)
        stage_macro_box = []
        for c in stage_raw:
            bx = c["bbox"]
            box_area = (bx[2] - bx[0]) * (bx[3] - bx[1])
            if box_area <= 0.30 * (W * H):
                stage_macro_box.append(c)

        # Gating Check 2: Horizon check (must be below camera horizon / road surface horizon)
        # Approximate road horizon from cy = 540
        v_horiz = 520.0
        stage_horizon = []
        for c in stage_macro_box:
            bx = c["bbox"]
            if bx[1] >= v_horiz - 20:
                stage_horizon.append(c)

        # Gating Check 3: Macro-mask artifact (>3% of screen)
        stage_macro_mask = []
        for c in stage_horizon:
            if c["pixel_area"] <= 0.03 * (W * H):
                stage_macro_mask.append(c)

        # Gating Check 4: Minimum pixel area (<25 px)
        stage_min_px = []
        for c in stage_macro_mask:
            if c["pixel_area"] >= 25:
                stage_min_px.append(c)

        # Gating Check 5: Previous script bug check:
        # Did the previous script reject cracks because disp_pred was <= 0.5?
        # Compute disparity on stereo pair
        gray_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
        disp_pred = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        stage_disp_rejected = []
        stage_disp_passed = []
        for c in stage_min_px:
            cnts_c, _ = cv2.findContours(c["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_c:
                stage_disp_rejected.append(c)
                continue
            c_cnt = max(cnts_c, key=cv2.contourArea)
            pts_c_2d = c_cnt[:, 0, :]
            u_c = pts_c_2d[:, 0]
            v_c = pts_c_2d[:, 1]
            d_c = disp_pred[v_c, u_c]
            if np.sum(d_c > 0.5) < 3:
                stage_disp_rejected.append(c)
            else:
                stage_disp_passed.append(c)

        print(f"  STEP 3 (GATING TRACE at conf>=0.03):")
        print(f"    1. Raw (conf>=0.03)        : {len(stage_raw)}")
        print(f"    2. After Macro-Box (>30%)   : {len(stage_macro_box)} (rejected {len(stage_raw)-len(stage_macro_box)})")
        print(f"    3. After Horizon (<500px)   : {len(stage_horizon)} (rejected {len(stage_macro_box)-len(stage_horizon)})")
        print(f"    4. After Macro-Mask (>3%)   : {len(stage_macro_mask)} (rejected {len(stage_horizon)-len(stage_macro_mask)})")
        print(f"    5. After Min-Pixel (<25px)  : {len(stage_min_px)} (rejected {len(stage_macro_mask)-len(stage_min_px)})")
        print(f"    6. Flawed Disparity Sampling: {len(stage_disp_passed)} passed vs {len(stage_disp_rejected)} REJECTED BY NOISY STEREO!")

        # -------------------------------------------------------------
        # STEP 4: CORRECT CRACK METROLOGY (ROAD-PLANE RAY SKELETONIZATION)
        # -------------------------------------------------------------
        # Road plane estimate from lower road surface:
        # Standard forward-facing camera road plane: aX + bY + cZ + d = 0
        # Normal approx [0, -1, 0], d approx 1.65m
        ref_road_plane = np.array([0.0, -1.0, 0.05, 1.65], dtype=np.float32)

        # Merge validated crack masks
        crack_union_mask = np.zeros((H, W), dtype=np.uint8)
        for c in stage_min_px:
            crack_union_mask = cv2.bitwise_or(crack_union_mask, c["mask"] * 255)

        crack_instances = crack_engine.analyze_cracks(
            crack_mask=crack_union_mask,
            road_plane=ref_road_plane,
            calib=calib,
            observable_road_area_m2=250.0
        )

        total_crack_len_m = sum(ci.length_m for ci in crack_instances)
        post_gating_count = len(crack_instances)
        print(f"  STEP 4 (CORRECT METROLOGY):")
        print(f"    Connected Crack Instances: {post_gating_count}")
        print(f"    Total Metric Crack Length: {total_crack_len_m:.2f} m")

        # Save final post-gating metrology image
        vis_final = img.copy()
        for ci in crack_instances:
            cv2.drawContours(vis_final, [cv2.findContours(ci.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, (0, 255, 255), 2)
            bx = ci.bbox_xyxy
            cv2.rectangle(vis_final, (bx[0], bx[1]), (bx[2], bx[3]), (0, 200, 255), 1)
            lbl = f"CRACK #{ci.track_id:02d} | Len: {ci.length_m:.2f}m"
            cv2.putText(vis_final, lbl, (bx[0], max(bx[1]-6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.putText(vis_final, f"POST-GATING ROAD-PLANE METROLOGY (Cracks: {post_gating_count}, Total Len: {total_crack_len_m:.2f}m)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"step4_final_metrology_frame_{f_idx:02d}.jpg"), vis_final)

        # Summary entry
        summary_table.append({
            "frame": f_idx,
            "filename": f_name,
            "raw_count": len(raw_cracks_640),
            "count_003": cnt_03,
            "count_005": cnt_05,
            "count_008": cnt_08,
            "count_010": cnt_10,
            "count_015": cnt_15,
            "post_gating_count": post_gating_count,
            "total_length_m": round(total_crack_len_m, 2),
            "max_confidence": round(max_conf_640, 3),
            "tiled_raw_count": len(raw_cracks_tiled),
            "tiled_max_conf": round(max_conf_tiled, 3),
            "tiled_count_003": cnt_tiled_03,
            "tiled_count_005": cnt_tiled_05
        })

    # Save summary JSON
    summary_json_path = out_dir / "crack_diagnostic_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_table, f, indent=2)

    # Print summary table
    print("\n" + "=" * 115)
    print("CRACK DIAGNOSTIC SUMMARY TABLE ACROSS REPRESENTATIVE FRAMES")
    print("=" * 115)
    header = f"{'Frame':^7} | {'Raw Count':^10} | {'@0.03':^7} | {'@0.05':^7} | {'@0.08':^7} | {'@0.10':^7} | {'@0.15':^7} | {'Post-Gate':^9} | {'Max Conf':^9} | {'Notes / Scene Description':^30}"
    print(header)
    print("-" * 115)

    for r in summary_table:
        f_i = r["frame"]
        note = "Clean Control" if f_i == 0 else "Minor Cracking" if f_i == 22 else "Obvious Alligator Cracks" if f_i in [41, 44, 45, 49] else "Severe Fractured Web Cracks"
        row = f"  {f_i:02d}   |    {r['raw_count']:4d}    |  {r['count_003']:4d}  |  {r['count_005']:4d}  |  {r['count_008']:4d}  |  {r['count_010']:4d}  |  {r['count_015']:4d}  |    {r['post_gating_count']:4d}   |   {r['max_confidence']:.3f}   | {note:<30}"
        print(row)
    print("=" * 115)


if __name__ == "__main__":
    main()
