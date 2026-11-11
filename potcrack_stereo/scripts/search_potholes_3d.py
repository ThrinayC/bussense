import subprocess
import json

queries = [
    "\"3D SBS\" pothole",
    "\"3D SBS\" \"bad road\"",
    "\"3D SBS\" \"rough road\"",
    "VR180 pothole",
    "VR180 \"bad road\"",
    "\"side by side\" pothole road",
    "\"stereo camera\" pothole road",
    "\"3D\" driving bad road potholes",
    "potholes stereo dashcam",
    "driving worst road 3D SBS"
]

results = []
seen = set()
for q in queries:
    cmd = ["uv", "tool", "run", "yt-dlp", f"ytsearch8:{q}", "--dump-json", "--flat-playlist"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    for line in p.stdout.strip().split("\n"):
        if not line: continue
        try:
            d = json.loads(line)
            vid = d.get("id")
            if vid and vid not in seen:
                seen.add(vid)
                results.append((vid, d.get("duration", 0), d.get("title", ""), q))
        except Exception:
            pass

print(f"Total results: {len(results)}")
for vid, dur, title, q in results:
    safe_title = title.encode('ascii', 'replace').decode('ascii')
    print(f"{vid} | {dur}s | {safe_title} | ({q})")
