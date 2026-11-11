import subprocess
import json

queries = [
    '"3D SBS" driving bumpy road',
    '"3D SBS" driving rough road',
    '"3D SBS" driving gravel',
    '"3D SBS" driving country road',
    '"3D SBS" driving off road',
    '"Side by side" 3D driving bumpy',
    'VR180 driving bumpy road',
    'VR180 driving dirt road',
    'stereoscopic driving bad road',
    '3D SBS road damage drive'
]

results = []
seen = set()

for q in queries:
    cmd = ["uv", "tool", "run", "yt-dlp", f"ytsearch6:{q}", "--dump-json", "--flat-playlist"]
    try:
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
    except Exception as e:
        print("Error on", q, e)

print(f"Total results: {len(results)}")
for vid, dur, title, q in results:
    safe = title.encode('ascii', 'replace').decode('ascii')
    print(f"{vid} | {dur}s | {safe} | ({q})")
