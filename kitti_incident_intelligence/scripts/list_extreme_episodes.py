import urllib.request
import json

url = "https://huggingface.co/api/datasets/Stary108/Extreme_Driving_Conditions_Dataset"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
data = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
siblings = [s.get("rfilename") for s in data.get("siblings", [])]

print("=== Critical Driving Episodes ===")
crit = [s for s in siblings if "critical" in s.lower()]
for c in crit[:25]:
    print(" ", c)

print("\n=== Emergency / Braking / Maneuver Episodes ===")
emerg = [s for s in siblings if any(k in s.lower() for k in ["emergency", "brak", "evasive", "avoid"])]
for e in emerg[:25]:
    print(" ", e)

print("\n=== Cut In / Lane Change Episodes ===")
lane = [s for s in siblings if any(k in s.lower() for k in ["cut", "lane_change", "overtake"])]
for l in lane[:25]:
    print(" ", l)

print(f"\nTotal: Critical={len(crit)}, Emergency={len(emerg)}, LaneChange={len(lane)}")
