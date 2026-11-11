#!/usr/bin/env python3
"""
scripts/populate_pothole_dataset.py
Extracts the 74 stereo frame pairs for data/pothole_long_stereo:
- Left frames: 1920x1080 from outputs/demo/pothole_long_stereo_demo.mp4
- Right frames: extracted from Q2 (0:540, 960:1920) of outputs/demo/pothole_long_stereo_debug.mp4,
  with 3 green epipolar lines and HUD badge cleanly restored via Telea inpainting,
  then upscaled to 1920x1080.
- sequence_meta.json generated for full compatibility.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "pothole_long_stereo"
LEFT_DIR = DATA_DIR / "left"
RIGHT_DIR = DATA_DIR / "right"

LEFT_DIR.mkdir(parents=True, exist_ok=True)
RIGHT_DIR.mkdir(parents=True, exist_ok=True)

demo_mp4 = PROJECT_ROOT / "outputs" / "demo" / "pothole_long_stereo_demo.mp4"
debug_mp4 = PROJECT_ROOT / "outputs" / "demo" / "pothole_long_stereo_debug.mp4"

if not demo_mp4.exists() or not debug_mp4.exists():
    raise FileNotFoundError("Source demo or debug mp4 not found in outputs/demo/")

cap_demo = cv2.VideoCapture(str(demo_mp4))
cap_debug = cv2.VideoCapture(str(debug_mp4))

n_frames_demo = int(cap_demo.get(cv2.CAP_PROP_FRAME_COUNT))
n_frames_debug = int(cap_debug.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames = min(n_frames_demo, n_frames_debug, 74)

print(f"Extracting {total_frames} stereo pairs to {DATA_DIR}...")

# Epipolar lines & badge inpaint mask for Q2 (540x960)
inpaint_mask = np.zeros((540, 960), dtype=np.uint8)
for ly in [135, 270, 405]:
    inpaint_mask[max(0, ly - 2):min(540, ly + 3), :] = 255
# Q2 badge
inpaint_mask[10:55, 10:380] = 255

sequence_meta = []

for idx in range(total_frames):
    ret_demo, frame_demo = cap_demo.read()
    ret_debug, frame_debug = cap_debug.read()
    if not ret_demo or not ret_debug:
        print(f"Warning: Reached end of video early at frame {idx}")
        break

    filename = f"frame_{idx:05d}.jpg"
    
    # Left frame: 1920x1080
    left_path = LEFT_DIR / filename
    cv2.imwrite(str(left_path), frame_demo, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Right frame: extract Q2 (0:540, 960:1920)
    q2 = frame_debug[0:540, 960:1920].copy()
    q2_clean = cv2.inpaint(q2, inpaint_mask, 3, cv2.INPAINT_TELEA)
    right_1080 = cv2.resize(q2_clean, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    
    right_path = RIGHT_DIR / filename
    cv2.imwrite(str(right_path), right_1080, [cv2.IMWRITE_JPEG_QUALITY, 95])

    sequence_meta.append({
        "frame_id": idx,
        "filename": filename,
        "session": "pothole_long_stereo_reel",
        "original_stem": f"frame_{idx:05d}"
    })

    if (idx + 1) % 10 == 0 or idx == total_frames - 1:
        print(f"  Extracted {idx + 1}/{total_frames} stereo pairs...")

cap_demo.release()
cap_debug.release()

meta_path = DATA_DIR / "sequence_meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(sequence_meta, f, indent=4)

print(f"\nSuccessfully populated {len(sequence_meta)} stereo pairs:")
print(f"  Left:  {LEFT_DIR}")
print(f"  Right: {RIGHT_DIR}")
print(f"  Meta:  {meta_path}")
