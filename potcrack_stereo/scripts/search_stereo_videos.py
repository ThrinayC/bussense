import subprocess
import json
import sys

queries = [
    "3D SBS driving potholes",
    "3D SBS rough road driving",
    "stereo dashcam potholes",
    "ZED stereo camera driving road",
    "stereo camera pothole detection",
    "driving on bad road 3D SBS",
    "pothole road test stereo camera",
    "side by side 3D road drive"
]

all_entries = []
seen_ids = set()

for q in queries:
    cmd = [
        "uv", "tool", "run", "yt-dlp",
        f"ytsearch8:{q}",
        "--dump-json",
        "--flat-playlist"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        for line in res.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                entry = json.loads(line)
                vid = entry.get("id")
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    all_entries.append({
                        "query": q,
                        "id": vid,
                        "title": entry.get("title"),
                        "duration": entry.get("duration"),
                        "url": entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
                        "description": entry.get("description", "")[:120]
                    })
            except Exception:
                pass
    except Exception as e:
        print(f"Error on {q}: {e}")

for i, e in enumerate(all_entries):
    print(f"[{i+1}] {e['id']} | {e['duration']}s | {e['title']}")
    print(f"    URL: {e['url']}")
    print(f"    Query: {e['query']}")
