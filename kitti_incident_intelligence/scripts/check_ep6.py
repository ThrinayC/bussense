import urllib.request
import json

url = "https://huggingface.co/api/datasets/Stary108/Extreme_Driving_Conditions_Dataset"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
data = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
siblings = [s.get("rfilename") for s in data.get("siblings", [])]

ep6_files = [s for s in siblings if "Emergency_Braking/episode_6" in s]
print(f"Files for Emergency_Braking/episode_6 ({len(ep6_files)}):")
for f in ep6_files[:30]:
    print(" ", f)

# Also check other Critical_Driving episodes
crit_dirs = set()
for s in siblings:
    if s.startswith("train/Critical_Driving/"):
        parts = s.split("/")
        if len(parts) >= 4:
            crit_dirs.add(f"{parts[2]}/{parts[3]}")
print("\nCritical Driving subcategories/episodes:")
for cd in sorted(list(crit_dirs))[:20]:
    print(" ", cd)
