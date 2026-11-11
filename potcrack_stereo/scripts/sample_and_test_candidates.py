import subprocess
import os
import glob
import cv2
import torch
from ultralytics import YOLO

os.makedirs("temp_candidates", exist_ok=True)

# Candidate video IDs
vids = {
    "zed_road_test": "TeYCVHm3u_g",
    "dashcam_3d_sbs": "0pDbV6IDwLE",
    "driving_full_sbs_16s": "RaoWS_yiF4g",
    "main_street_sbs_23s": "lf8nYOKtw4E",
    "gopro_dashcam_sbs": "Acgt9Ar4NTg"
}

yolo = YOLO("checkpoints/segmentation/best_pothole_seg.pt")

for name, vid in vids.items():
    out_path = f"temp_candidates/{name}.mp4"
    if not os.path.exists(out_path):
        print(f"\n--- Downloading sample of {name} ({vid}) ---")
        # Download at low/medium resolution (720p or 480p) to be fast
        cmd = [
            "uv", "tool", "run", "yt-dlp",
            "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "--download-sections", "*00:00-00:25",  # first 25 seconds
            "-o", out_path,
            f"https://www.youtube.com/watch?v={vid}"
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        print("yt-dlp stdout:", p.stdout[-300:] if p.stdout else "")
        print("yt-dlp stderr:", p.stderr[-300:] if p.stderr else "")
    
    if os.path.exists(out_path):
        cap = cv2.VideoCapture(out_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n[{name}] {w}x{h} @ {fps:.1f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")
        
        # Sample 5 frames across the video
        pothole_hits = 0
        samples = []
        for fi in [5, total_frames//4, total_frames//2, 3*total_frames//4, max(5, total_frames-5)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            # If SBS, split left and right
            left = frame[:, :w//2]
            right = frame[:, w//2:]
            
            # Run YOLO on left frame
            res = yolo(left, conf=0.25, verbose=False)[0]
            num_det = len(res.boxes) if res.boxes is not None else 0
            if num_det > 0:
                pothole_hits += num_det
            samples.append((fi, left, right, num_det))
        
        cap.release()
        print(f"  Pothole detections across 5 sample frames: {pothole_hits}")
        # Save a montage sample
        if samples:
            fi, left, right, _ = samples[min(2, len(samples)-1)]
            cv2.imwrite(f"temp_candidates/{name}_sample_left.jpg", left)
            cv2.imwrite(f"temp_candidates/{name}_sample_right.jpg", right)
