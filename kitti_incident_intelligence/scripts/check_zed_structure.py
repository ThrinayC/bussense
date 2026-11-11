import urllib.request
import json

url = "https://huggingface.co/api/datasets/Stary108/Extreme_Driving_Conditions_Dataset"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
data = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
siblings = [s.get("rfilename") for s in data.get("siblings", [])]

ep6_files = [s for s in siblings if "Emergency_Braking/episode_6" in s]

zed_dirs = set()
img_dirs = set()
for f in ep6_files:
    if "/zed/" in f:
        zed_dirs.add(f.split("/zed/")[1].split("/")[0])
    if "/images/" in f:
        img_dirs.add(f.split("/images/")[1].split("/")[0])

print("Subdirs in /zed/:", zed_dirs)
print("Subdirs in /images/:", img_dirs)
