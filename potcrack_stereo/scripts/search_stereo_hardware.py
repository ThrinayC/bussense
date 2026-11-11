import subprocess
import json

queries = [
    '"OAK-D" road',
    '"OAK-D" pothole',
    '"ZED 2" road driving',
    '"ZED 2" pothole',
    '"ZED" road damage',
    '"Insta360 EVO" driving',
    '"QooCam" driving rough',
    '"GoPro 3D" driving road',
    'VR180 driving rough',
    'stereo camera vehicle potholes',
    '"stereo vision" road potholes',
    'stereoscopic vehicle bad road'
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
with open('outputs/hardware_search_results.txt', 'w', encoding='utf-8') as f:
    for vid, dur, title, q in results:
        f.write(f"{vid} | {dur}s | {title} | ({q})\n")
        safe = title.encode('ascii', 'replace').decode('ascii')
        print(f"{vid} | {dur}s | {safe}")
