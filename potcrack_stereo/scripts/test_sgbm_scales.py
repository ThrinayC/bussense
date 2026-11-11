#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stereo_depth import get_stereo_matcher

data_dir = PROJECT_ROOT / "data" / "pothole_long_stereo"
lf = cv2.imread(str(data_dir / "left" / "frame_00015.jpg"))
rf = cv2.imread(str(data_dir / "right" / "frame_00015.jpg"))

matcher = get_stereo_matcher("sgbm")

# Full 1080p
disp_1080 = matcher.estimate_disparity(lf, rf)
valid_1080 = disp_1080 > 0.5
print(f"1080p SGBM: Valid fraction = {np.mean(valid_1080):.1%}, Median disp = {np.median(disp_1080[valid_1080]):.2f} px")

# 540p (960x540)
lf_540 = cv2.resize(lf, (960, 540))
rf_540 = cv2.resize(rf, (960, 540))
disp_540 = matcher.estimate_disparity(lf_540, rf_540)
valid_540 = disp_540 > 0.5
print(f"540p SGBM : Valid fraction = {np.mean(valid_540):.1%}, Median disp = {np.median(disp_540[valid_540]):.2f} px")

# Compare depth maps:
# For 1080p: fx = 1050, B = 0.12 => Z = (1050 * 0.12) / disp_1080
# For 540p: fx = 525, B = 0.12 => Z = (525 * 0.12) / disp_540
z_1080 = np.where(valid_1080, (1050.0 * 0.12) / disp_1080, 0.0)
z_540 = np.where(valid_540, (525.0 * 0.12) / disp_540, 0.0)

print(f"1080p Depth (valid road): min={np.min(z_1080[valid_1080]):.2f}m, med={np.median(z_1080[valid_1080]):.2f}m, max={np.max(z_1080[valid_1080]):.2f}m")
print(f"540p Depth  (valid road): min={np.min(z_540[valid_540]):.2f}m, med={np.median(z_540[valid_540]):.2f}m, max={np.max(z_540[valid_540]):.2f}m")
