import subprocess
import json

queries = [
    'VR180 driving "bad road"',
    'VR180 driving "potholes"',
    'VR180 driving "India"',
    'VR180 car ride bumpy',
    '"3D SBS" driving India',
    '"3D SBS" driving "bad road"',
    '"3D SBS" driving "potholes"',
    'VR180 driving village road',
    'VR180 motorcycle bad road potholes',
    '3D SBS motorcycle potholes'
]

candidates = []
seen = set()
for q in queries:
    cmd = ['uv', 'tool', 'run', 'yt-dlp', f'ytsearch6:{q}', '--dump-json', '--flat-playlist']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        for line in p.stdout.strip().split('\n'):
            if not line: continue
            try:
                d = json.loads(line)
                vid = d.get('id')
                if vid and vid not in seen:
                    seen.add(vid)
                    candidates.append((vid, d.get('duration', 0), d.get('title', ''), q))
            except Exception:
                pass
    except Exception as e:
        print(f"Error on {q}: {e}")

print(f"Total candidates found: {len(candidates)}")
for vid, dur, title, q in candidates:
    safe = title.encode('ascii', 'replace').decode('ascii')
    print(f"{vid} | {dur}s | {safe} | ({q})")
