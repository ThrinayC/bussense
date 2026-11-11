import json
import urllib.request
import subprocess
import os

print("--- Checking Hugging Face khoa-na/pothole-gps-localization-demos ---")
try:
    req = urllib.request.Request(
        "https://huggingface.co/api/datasets/khoa-na/pothole-gps-localization-demos/tree/main",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        tree = json.loads(resp.read().decode("utf-8"))
        print(f"Found {len(tree)} entries in HF dataset:")
        for item in tree:
            print("  ", item.get("path"), item.get("size"))
except Exception as e:
    print(f"HF query error: {e}")

print("\n--- Searching YouTube for 3D SBS road/pothole driving ---")
queries = [
    "3D SBS driving potholes",
    "3D SBS rough road driving",
    "stereo dashcam road potholes",
    "ZED stereo camera road driving",
    "3D side by side car drive pothole",
    "Stereo camera driving bad road"
]

results = []
seen = set()
for q in queries:
    cmd = ["uv", "tool", "run", "yt-dlp", f"ytsearch6:{q}", "--dump-json", "--flat-playlist"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        for line in p.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                vid = data.get("id")
                if vid and vid not in seen:
                    seen.add(vid)
                    results.append({
                        "id": vid,
                        "title": data.get("title"),
                        "duration": data.get("duration"),
                        "url": data.get("url") or f"https://www.youtube.com/watch?v={vid}"
                    })
            except Exception:
                pass
    except Exception as e:
        print(f"Error searching {q}: {e}")

with open("outputs/stereo_video_candidates.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Found {len(results)} candidate videos saved to outputs/stereo_video_candidates.json")
for r in results[:10]:
    # safe print
    title = r['title'].encode('ascii', 'replace').decode('ascii')
    print(f"  {r['id']} | {r['duration']}s | {title}")
