import urllib.request
import json

url = "https://huggingface.co/api/datasets/Stary108/Extreme_Driving_Conditions_Dataset"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
data = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
siblings = [s.get("rfilename") for s in data.get("siblings", [])]

ep6_files = [s for s in siblings if "Emergency_Braking/episode_6/" in s]
subdirs = set()
for f in ep6_files:
    rel = f.replace("train/Critical_Driving/Emergency_Braking/episode_6/", "")
    parts = rel.split("/")
    if len(parts) > 1:
        subdirs.add(parts[0])
    else:
        subdirs.add(parts[0])
print("Subdirectories / files in episode_6:")
for s in sorted(list(subdirs)):
    print(" ", s)

for f in ep6_files[:10]:
    print(" Sample file:", f)
