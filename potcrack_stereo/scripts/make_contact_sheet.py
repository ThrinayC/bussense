import cv2
import json
import shutil
from pathlib import Path
import numpy as np
from ultralytics import YOLO

out_dir = Path("data/pothole_long_stereo")
with open(out_dir / "sequence_meta.json", "r") as f:
    meta = json.load(f)

yolo = YOLO("checkpoints/segmentation/best_pothole_seg.pt")

# Sample diverse indices across the 74 frames
sample_indices = [3, 14, 25, 34, 46, 60]

rows = []
for idx in sample_indices:
    m = meta[idx]
    lf_path = out_dir / "left" / m["filename"]
    rf_path = out_dir / "right" / m["filename"]
    
    left = cv2.imread(str(lf_path))
    right = cv2.imread(str(rf_path))
    
    # Verify disparity exists
    diff = float(np.mean(np.abs(left.astype(float) - right.astype(float))))
    print(f"Frame {idx:02d} ({m['session']} - {m['original_stem']}): L/R pixel diff = {diff:.2f}")
    
    # YOLO segmentation on left
    res = yolo(left, conf=0.25, verbose=False)[0]
    n_det = len(res.boxes) if res.boxes is not None else 0
    
    left_ann = left.copy()
    if res.masks is not None and len(res.masks) > 0:
        for seg, box in zip(res.masks.xy, res.boxes):
            poly = seg.astype(np.int32)
            cv2.fillPoly(left_ann, [poly], (0, 140, 255))
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(left_ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(left_ann, f"Pothole {box.conf[0]:.2f}", (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.addWeighted(left_ann, 0.65, left, 0.35, 0, left_ann)
    
    # Scale down for contact sheet
    h, w = left.shape[:2]
    disp_w, disp_h = 480, int(480 * h / w)
    l_small = cv2.resize(left, (disp_w, disp_h))
    r_small = cv2.resize(right, (disp_w, disp_h))
    a_small = cv2.resize(left_ann, (disp_w, disp_h))
    
    # Epipolar reference line
    cv2.line(l_small, (0, disp_h // 2), (disp_w, disp_h // 2), (0, 255, 255), 1)
    cv2.line(r_small, (0, disp_h // 2), (disp_w, disp_h // 2), (0, 255, 255), 1)
    
    # Header labels
    cv2.putText(l_small, f"F{idx:02d} [{m['session']}] LEFT", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(r_small, f"F{idx:02d} RIGHT (Disparity diff={diff:.1f})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(a_small, f"F{idx:02d} YOLO POTHOLES: {n_det}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if n_det > 0 else (180, 180, 180), 2)
    
    row = np.hstack([l_small, r_small, a_small])
    rows.append(row)

contact_sheet = np.vstack(rows)
Path("outputs/demo").mkdir(parents=True, exist_ok=True)
out_path = Path("outputs/demo/pothole_candidates_contact_sheet.png")
cv2.imwrite(str(out_path), contact_sheet)

brain_dir = Path(r"C:\Users\Integra\.gemini\antigravity\brain\bdb57915-d8a4-47a7-b023-aa7ea297fbb4")
shutil.copy2(str(out_path), brain_dir / "pothole_candidates_contact_sheet.png")
print("Saved contact sheet successfully!")
