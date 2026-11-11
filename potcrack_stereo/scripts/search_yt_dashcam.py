import subprocess
import json

queries = [
    '"3D SBS" dashcam',
    '"side by side" 3D dashcam',
    'stereoscopic dashcam road',
    'ZED 2 driving road',
    'ZED 2i driving',
    'Stereolabs ZED driving test',
    'stereo camera driving bumpy road',
    '3D SBS driving street',
    '3D SBS road pothole',
    'dashcam 3D SBS',
    'VR180 driving rough road'
]

candidates = []
seen = set()
for q in queries:
    cmd = ['uv', 'tool', 'run', 'yt-dlp', f'ytsearch6:{q}', '--dump-json', '--flat-playlist']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        for line in p.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                d = json.loads(line)
                vid = d.get('id')
                if vid and vid not in seen:
                    seen.add(vid)
                    candidates.append((vid, d.get('duration', 0), d.get('title', '')))
            except Exception:
                pass
    except Exception as e:
        print(f"Error on query {q}: {e}")

with open('outputs/yt_dashcam_candidates.txt', 'w', encoding='utf-8') as f:
    for vid, dur, title in candidates:
        f.write(f"{vid} | {dur}s | {title}\n")

print(f"Done, found {len(candidates)} candidates")
