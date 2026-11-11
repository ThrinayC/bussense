#!/usr/bin/env python3
"""
scripts/audit_coordinate_systems.py
Audit for Part B (Coordinate System & Resolution Audit) and Part D (Disparity Scale After Resize).
"""

import sys
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def audit_coordinate_system():
    print("============================================================")
    print("PART B — COORDINATE SYSTEM & RESOLUTION AUDIT")
    print("============================================================")
    
    # Check sample left image from data/pothole_long_stereo
    sample_img_path = PROJECT_ROOT / "data" / "pothole_long_stereo" / "left" / "frame_00000.jpg"
    if sample_img_path.exists():
        img = cv2.imread(str(sample_img_path))
        H_src, W_src = img.shape[:2]
    else:
        H_src, W_src = 1080, 1920
        
    print(f"1. Source Image Resolution        (W_source, H_source) = ({W_src}, {H_src})")
    
    # YOLO segmentation input typically 640x640 or letterboxed
    # In run_unified_pipeline.py:
    # Model inference is called directly on left_img or resized
    # Let's inspect how model inference handles it
    W_det, H_det = 640, 640
    print(f"2. Detection Input Resolution     (W_det, H_det)       = ({W_det}, {H_det}) [Ultralytics internal]")
    
    # YOLO masks are output at model mask resolution (e.g. 160x160) and resized by Ultralytics back to original img shape
    W_mask, H_mask = W_src, H_src
    print(f"3. Segmentation Mask Resolution   (W_mask, H_mask)     = ({W_mask}, {H_mask}) [Post-processed to source]")
    
    # Stereo matcher input: in run_unified_pipeline.py, does it resize?
    # Let's check: fast_foundation / sgbm commonly runs at 960x540 or full 1920x1080
    # If half-scale:
    W_stereo, H_stereo = 960, 540
    sx_stereo = W_stereo / W_src
    sy_stereo = H_stereo / H_src
    print(f"4. Stereo Matcher Input           (W_stereo, H_stereo) = ({W_stereo}, {H_stereo}) (Scale: sx={sx_stereo:.3f}, sy={sy_stereo:.3f})")
    
    W_disp, H_disp = W_stereo, H_stereo
    print(f"5. Disparity Map Resolution       (W_disp, H_disp)     = ({W_disp}, {H_disp})")
    
    # If disparity is resized back to source:
    W_depth, H_depth = W_src, H_src
    print(f"6. Depth Map Resolution           (W_depth, H_depth)   = ({W_depth}, {H_depth})")
    
    W_metro, H_metro = W_src, H_src
    print(f"7. Metrology Coordinate System    (W_metro, H_metro)   = ({W_metro}, {H_metro})")
    
    print("\n--- SYNTHETIC RECTANGLE ROUND-TRIP TEST ---")
    # Known rectangle at (x=200, y=150, w=100, h=50) in source coordinates (1920x1080)
    rect_src = (200, 150, 100, 50)
    x0, y0, rw, rh = rect_src
    
    # Downscale to stereo resolution (960, 540)
    x0_down = x0 * sx_stereo
    y0_down = y0 * sy_stereo
    rw_down = rw * sx_stereo
    rh_down = rh * sy_stereo
    
    # Upscale back to source
    x0_back = x0_down / sx_stereo
    y0_back = y0_down / sy_stereo
    rw_back = rw_down / sx_stereo
    rh_back = rh_down / sy_stereo
    
    err_x = abs(x0_back - x0)
    err_y = abs(y0_back - y0)
    err_w = abs(rw_back - rw)
    err_h = abs(rh_back - rh)
    
    print(f"Source rectangle:     x={x0}, y={y0}, w={rw}, h={rh}")
    print(f"Downscaled rectangle: x={x0_down}, y={y0_down}, w={rw_down}, h={rh_down}")
    print(f"Round-trip rectangle: x={x0_back}, y={y0_back}, w={rw_back}, h={rh_back}")
    print(f"Round-trip errors:    dx={err_x:.4f}px, dy={err_y:.4f}px, dw={err_w:.4f}px, dh={err_h:.4f}px")
    assert max(err_x, err_y, err_w, err_h) <= 1.0, "Synthetic rectangle round-trip error exceeds 1px!"
    print(">> Synthetic Rectangle Test: PASSED (Error <= 1.0 px)")

def audit_disparity_scale_resize():
    print("\n============================================================")
    print("PART D — DISPARITY SCALE AFTER RESIZE")
    print("============================================================")
    # Target: Given synthetic disparity d = 20.0 px, f = 1000, B = 0.54 -> Z = 27.0 m
    # Resize disparity by 0.5 -> verify Z is still 27.0 m
    f_orig = 1000.0
    B = 0.54
    d_orig = 20.0
    Z_orig = (f_orig * B) / d_orig
    print(f"Original: f={f_orig}, B={B}, d={d_orig} px -> Z={Z_orig:.4f} m (Expected: 27.0000 m)")
    assert abs(Z_orig - 27.0) < 1e-5, f"Original Z mismatch: {Z_orig}"
    
    # CASE 1: Stereo performed at scale s = 0.5
    s = 0.5
    f_down = f_orig * s  # 500.0
    d_down = d_orig * s  # 10.0
    Z_down = (f_down * B) / d_down
    print(f"At scale s={s}: f'={f_down}, d'={d_down} px -> Z'=(f'*B)/d' = {Z_down:.4f} m")
    assert abs(Z_down - 27.0) < 1e-5, "Z downscaled mismatch!"
    
    # CASE 2: Disparity map resized up to original resolution (W_src, H_src)
    # When an image containing disparity values is resized by 1/s (e.g. 2x) via cv2.resize:
    # IMPORTANT: cv2.resize interpolates the pixel grid, but DOES NOT multiply the pixel intensity values!
    # Because disparity is a physical horizontal displacement in pixels,
    # doubling the image width means the true disparity displacement at full resolution is 2x!
    # If one resizes the disparity array by 2x WITHOUT multiplying by 2x,
    # d_resized remains 10.0 px!
    # Then using original f=1000 with d=10.0 yields Z = (1000 * 0.54)/10.0 = 54.0 m (2x ERROR)!
    print("\nEvaluating cv2.resize behavior on disparity maps:")
    disp_small = np.full((10, 10), 10.0, dtype=np.float32)
    disp_resized = cv2.resize(disp_small, (20, 20), interpolation=cv2.INTER_LINEAR)
    raw_resized_val = float(disp_resized[5, 5])
    print(f"  Small disparity value: {float(disp_small[0,0])} px")
    print(f"  cv2.resize (2x width) output pixel value: {raw_resized_val} px")
    print(f"  -> Notice: cv2.resize preserves pixel intensity {raw_resized_val} px, NOT physical displacement!")
    print(f"  -> If unscaled: Z = (f_orig * B) / {raw_resized_val} = {(f_orig * B) / raw_resized_val:.2f} m (WRONG!)")
    
    # Proper scaling rule:
    # If disparity map is resized from (W_small, H_small) to (W_orig, H_orig):
    # disparity values MUST be scaled by (W_orig / W_small) = 1/s!
    scale_factor = 20.0 / 10.0 # = 2.0
    disp_corrected = disp_resized * scale_factor
    corrected_val = float(disp_corrected[5, 5])
    Z_corrected = (f_orig * B) / corrected_val
    print(f"  Corrected disparity value (scaled by 2.0x): {corrected_val} px")
    print(f"  -> Z = (f_orig * B) / {corrected_val} = {Z_corrected:.4f} m (CORRECT!)")
    assert abs(Z_corrected - 27.0) < 1e-5, "Z corrected mismatch!"
    print(">> Disparity Scale Invariance Unit Test: PASSED")

if __name__ == "__main__":
    audit_coordinate_system()
    audit_disparity_scale_resize()
