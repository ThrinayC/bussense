import os
import sys
import json
import cv2

from bussense_unified_orchestrator import BusSenseUnifiedOrchestrator
from kitti_incident_intelligence.scripts.calibration import KittiCalibration

kitti_calib = KittiCalibration(
    cam_to_cam_file=r'kitti_incident_intelligence/data/kitti/2011_09_26_calib/calib_cam_to_cam.txt',
    velo_to_cam_file=r'kitti_incident_intelligence/data/kitti/2011_09_26_calib/calib_velo_to_cam.txt'
)
orch = BusSenseUnifiedOrchestrator(calib=kitti_calib)

left_dir = r'kitti_incident_intelligence/data/kitti/2011_09_26/2011_09_26_drive_0002_sync/image_02/data'
right_dir = r'kitti_incident_intelligence/data/kitti/2011_09_26/2011_09_26_drive_0002_sync/image_03/data'
fnames = sorted(os.listdir(left_dir))[:5]

for i, fn in enumerate(fnames):
    l_img = cv2.imread(os.path.join(left_dir, fn))
    r_img = cv2.imread(os.path.join(right_dir, fn))
    state, arts = orch.process_stereo_frame(l_img, r_img, frame_id=i, timestamp=i*0.1, ego_speed_mps=5.0)
    print(f"Frame {i}: Vehicles={len(state.vehicles)}, VRUs={len(state.vru)}, Flow={state.traffic['flow_rate_vpm']} vpm, TrafficState={state.traffic['traffic_state']}, JamRisk={state.traffic['jam_risk']}, RoadCond={state.road['road_condition']}")

print("\n--- SAMPLE UNIFIED FRAME STATE JSON ---")
print(json.dumps(state.to_dict(), indent=2))
print("\nOrchestrator test succeeded!")
