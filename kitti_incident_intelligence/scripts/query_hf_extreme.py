import urllib.request
import json

url = "https://huggingface.co/api/datasets/Stary108/Extreme_Driving_Conditions_Dataset"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    data = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
    print("Dataset ID:", data.get("id"))
    print("Description / tags:", data.get("tags"))
    siblings = data.get("siblings", [])
    print(f"Total files in repo: {len(siblings)}")
    for s in siblings[:30]:
        print("  -", s.get("rfilename"))
except Exception as e:
    print("Error querying HF API:", e)
