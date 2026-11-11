import urllib.request
import json

url = "https://huggingface.co/api/datasets/Stary108/Extreme_Driving_Conditions_Dataset"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
data = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
siblings = [s.get("rfilename") for s in data.get("siblings", [])]

ep6_images = [s for s in siblings if "Emergency_Braking/episode_6/images" in s]
ep6_zed = [s for s in siblings if "Emergency_Braking/episode_6/zed" in s]

print(f"Images in episode_6 ({len(ep6_images)}):")
for f in ep6_images[:10]:
    print(" ", f)

print(f"Zed files in episode_6 ({len(ep6_zed)}):")
for f in ep6_zed[:10]:
    print(" ", f)
