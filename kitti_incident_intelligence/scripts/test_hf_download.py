import urllib.request
import json
import os

url = "https://huggingface.co/datasets/Stary108/Extreme_Driving_Conditions_Dataset/resolve/main/train/Critical_Driving/Emergency_Braking/episode_6/calibration.json"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    content = urllib.request.urlopen(req).read().decode('utf-8')
    data = json.loads(content)
    print("Successfully downloaded calibration.json!")
    print("Keys in calibration.json:", list(data.keys()))
    if "sensors" in data:
        print("Sensors:", list(data["sensors"].keys()))
    elif "camera" in data:
        print("Camera:", list(data["camera"].keys()))
    else:
        print("Sample calibration data:", json.dumps(data, indent=2)[:500])
except Exception as e:
    print("Download failed:", e)
