import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
from scripts.run_combined_pipeline import BusSensePerceptionPipeline

cap = cv2.VideoCapture('temp_candidates/main_street_sbs.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("Failed to read frame")
    exit(1)

w = frame.shape[1] // 2
left = frame[:, :w]
right = frame[:, w:]

calib = {'fx': 457.0, 'fy': 457.0, 'cx': 320.0, 'cy': 180.0, 'baseline': 0.12}
pipeline = BusSensePerceptionPipeline(calib=calib, stereo_model='sgbm')
out = pipeline.process_frame(left, right, frame_index=0)

print("=== BusSense Perception Pipeline Smoke Test Success! ===")
print("Disparity shape:", out["disparity"].shape)
print("Tracked Potholes:", len(out["potholes"]))
for p in out["potholes"][:2]:
    print(f"  Pothole #{p['track_id']}: Depth={p['depth_p95_cm']} cm, Area={p['area_m2']} m2, Quality={p['quality']}")

print("Tracked Vehicles:", len(out["traffic"]["vehicles"]))
for v in out["traffic"]["vehicles"][:2]:
    print(f"  Vehicle {v.get('display_id')}: Class={v.get('class')}, Dist={v.get('distance_m')} m, 3D=({v.get('X_m')}, {v.get('Y_m')}, {v.get('Z_m')}) m, Status={v.get('status')}")

print("Ego Metrics:", out["traffic"]["ego_metrics"])
print("Observable Road Area:", out["density"]["observable_road_area_m2"], "m2")
print("Traffic Density:", out["density"]["traffic_density_veh_km2"], "veh/km2 | Status:", out["density"]["density_status"])
print("Pipeline Timings:", out["runtime_ms"])
